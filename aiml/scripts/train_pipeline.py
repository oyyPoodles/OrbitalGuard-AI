# -*- coding: utf-8 -*-
import io as _io, sys as _sys
_sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
OrbitalGuard AI - Master Training Pipeline
===========================================
Orchestrates the full AIML training sequence:

  [0] Preprocessing   → TLE → normalized sequences (.npy)
  [1] LSTM Training   → Real TLE sequences → hybrid_lstm.pth
  [2] LSTM Fine-tune  → Adapt FC head on real data
  [3] LSTM Evaluation → RMSE / MAE on test split
  [4] XGBoost         → Risk classifier training + CV + evaluation
  [5] PPO             → Curriculum training + evaluation
  [6] Generate Plots  → confusion matrix, training curves, rewards

Usage:
  python aiml/scripts/train_pipeline.py
  python aiml/scripts/train_pipeline.py --skip-preprocess
  python aiml/scripts/train_pipeline.py --skip-ppo
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import time

# ─── Path Alignment ──────────────────────────────────────
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT    = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(AIML_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, 'backend')

for p in [AIML_ROOT, BACKEND_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocessing.tle_preprocessor import TLEPreprocessor
from prediction.lstm_model import train_on_real_data, fine_tune, evaluate, load_hybrid_model
from collision.risk_model import RiskClassifier
from avoidance.ppo_agent import PPOAvoidanceAgent, HAS_SB3

MODELS_DIR = os.path.join(AIML_ROOT, 'models')
DATA_DIR   = os.path.join(AIML_ROOT, 'data')
OUTPUT_DIR = os.path.join(AIML_ROOT, 'output')


# ────────────────────────────────────────────────────────
def header(step: str, total: int, current: int):
    bar = "─" * 58
    print(f"\n{bar}")
    print(f"  [{current}/{total}] {step}")
    print(f"{bar}")


def elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{s:.1f}s"


# ────────────────────────────────────────────────────────
def run_pipeline(args):
    t_global = time.time()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_steps = 6
    metrics     = {}

    print("\n" + "=" * 60)
    print("  ORBITALGUARD AI — MASTER TRAINING PIPELINE")
    print("=" * 60)

    # ── 0. Preprocessing ─────────────────────────────────
    if not args.skip_preprocess:
        header("TLE Preprocessing", total_steps, 0)
        t0 = time.time()
        preprocessor = TLEPreprocessor(
            max_objects=args.max_objects,
            n_propagation_steps=50
        )
        preprocessor.run()
        print(f"  ✅ Preprocessing done ({elapsed(t0)})")
    else:
        print("\n  [0] Preprocessing — SKIPPED (--skip-preprocess)")
        # Verify sequences exist
        for k in ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']:
            path = os.path.join(DATA_DIR, f'sequence_{k}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Remove --skip-preprocess to regenerate."
                )

    # ── 1. LSTM Training ──────────────────────────────────
    header("LSTM — Train on Real TLE Sequences", total_steps, 1)
    t0 = time.time()
    lstm_path = os.path.join(MODELS_DIR, 'hybrid_lstm.pth')

    lstm_model = train_on_real_data(
        data_dir   = DATA_DIR,
        save_path  = lstm_path,
        epochs     = args.lstm_epochs,
        batch_size = 256,
        lr         = 1e-3,
        patience   = 8
    )
    print(f"  ✅ LSTM training done ({elapsed(t0)})")

    # ── 2. LSTM Fine-tuning ───────────────────────────────
    header("LSTM — Fine-tune FC Head", total_steps, 2)
    t0 = time.time()
    lstm_model = fine_tune(
        model     = lstm_model,
        data_dir  = DATA_DIR,
        save_path = lstm_path,
        epochs    = 15,
        lr        = 3e-4
    )
    print(f"  ✅ Fine-tuning done ({elapsed(t0)})")

    # ── 3. LSTM Evaluation ────────────────────────────────
    header("LSTM — Evaluate on Test Split", total_steps, 3)
    lstm_metrics = evaluate(lstm_model, data_dir=DATA_DIR)
    metrics['lstm'] = lstm_metrics
    print(f"  ✅ LSTM RMSE: {lstm_metrics['rmse_km']:.4f} km")

    # ── 4. XGBoost ────────────────────────────────────────
    header("XGBoost — Train + Evaluate Risk Classifier", total_steps, 4)
    t0 = time.time()
    xgb_path = os.path.join(MODELS_DIR, 'xgb_risk.pkl')

    # Force retrain
    if os.path.exists(xgb_path):
        os.remove(xgb_path)

    rc = RiskClassifier(model_path=xgb_path, use_real_data=True)
    xgb_metrics = rc.evaluate()
    metrics['xgboost'] = xgb_metrics

    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    rc.feature_importance(save_path=fi_path)
    print(f"  ✅ XGBoost done ({elapsed(t0)})")

    # ── 5. PPO ────────────────────────────────────────────
    if not args.skip_ppo:
        header("PPO — Curriculum Training + Evaluation", total_steps, 5)
        t0 = time.time()

        if HAS_SB3:
            ppo_path = os.path.join(MODELS_DIR, 'ppo_avoidance.zip')
            agent    = PPOAvoidanceAgent(model_path=ppo_path)
            agent.train(total_timesteps=args.ppo_steps, use_curriculum=True)
            ppo_metrics      = agent.test(n_episodes=100)
            metrics['ppo']   = ppo_metrics
            print(f"  ✅ PPO done ({elapsed(t0)}) | Success: {ppo_metrics['success_rate']:.1%}")
        else:
            print("  ⚠  stable-baselines3 not installed — skipping PPO")
            print("     Install: pip install stable-baselines3[extra]")
    else:
        print("\n  [5] PPO Training — SKIPPED (--skip-ppo)")

    # ── 6. Generate Plots ─────────────────────────────────
    header("Generate Metric Plots", total_steps, 6)
    for script in ['generate_ml_metrics.py', 'extract_paper_metrics.py',
                   'generate_presentation_plots.py']:
        script_path = os.path.join(SCRIPTS_DIR, script)
        if os.path.exists(script_path):
            try:
                subprocess.run([sys.executable, script_path], check=True)
                print(f"  ✅ {script}")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠  {script} failed: {e}")

    # ── Summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {elapsed(t_global)}")
    print("=" * 60)

    if 'lstm' in metrics:
        print(f"  LSTM   RMSE : {metrics['lstm']['rmse_km']:.4f} km")
    if 'xgboost' in metrics:
        print(f"  XGBoost F1  : {metrics['xgboost']['f1_weighted']:.4f}")
        print(f"  XGBoost AUC : {metrics['xgboost']['roc_auc']:.4f}")
    if 'ppo' in metrics:
        print(f"  PPO Success : {metrics['ppo']['success_rate']:.1%}")

    print("\n  Models saved to:", MODELS_DIR)
    print("  Plots  saved to:", OUTPUT_DIR)
    print("=" * 60)

    return metrics


# ────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='OrbitalGuard AI — Master Training Pipeline'
    )
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip TLE preprocessing (use existing .npy sequences)')
    parser.add_argument('--skip-ppo',        action='store_true',
                        help='Skip PPO training')
    parser.add_argument('--max-objects',     type=int, default=2000,
                        help='Max TLE objects to preprocess (default: 2000)')
    parser.add_argument('--lstm-epochs',     type=int, default=60,
                        help='Max LSTM training epochs (default: 60)')
    parser.add_argument('--ppo-steps',       type=int, default=50000,
                        help='PPO total training timesteps (default: 50000)')
    args = parser.parse_args()

    run_pipeline(args)
