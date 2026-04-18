"""
OrbitalGuard AI — Master ML Training Pipeline
"""
import os
import sys
import subprocess

# ─── Path Alignment ───────────────────────────────────────
# Current file is aiml/scripts/train_all_models.py
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(AIML_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, 'backend')

# Add backend and aiml roots to sys.path
if BACKEND_ROOT not in sys.path: sys.path.insert(0, BACKEND_ROOT)
if AIML_ROOT not in sys.path: sys.path.insert(0, AIML_ROOT)

from prediction.lstm_model import train_hybrid_model
from collision.risk_model import RiskClassifier
from avoidance.ppo_agent import PPOAvoidanceAgent, HAS_SB3

def train_all():
    print("==================================================")
    print("[INITIALIZING] PROTOCOL: ORBITALGUARD AI TRAINING PIPELINE")
    print("==================================================\n")

    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Train Hybrid LSTM
    print("--- [1/3] Training Hybrid LSTM Trajectory Model ---")
    train_hybrid_model(save_path=os.path.join(MODELS_DIR, 'hybrid_lstm.pth'), epochs=30, seq_len=10)
    print("\n")

    # 2. Train XGBoost Classifier
    print("--- [2/3] Training XGBoost Risk Classifier ---")
    rc = RiskClassifier(model_path=os.path.join(MODELS_DIR, 'xgb_risk.pkl'))
    rc.train_and_save()
    print("\n")

    # 3. Train PPO Avoidance Agent (requires stable-baselines3)
    print("--- [3/3] Training PPO RL Avoidance Agent ---")
    if HAS_SB3:
        agent = PPOAvoidanceAgent(model_path=os.path.join(MODELS_DIR, 'ppo_avoidance.zip'))
        agent.train(total_timesteps=15000)
    else:
        print("[Warning] stable-baselines3 not installed. Skipping PPO training.")
        print("Install via: pip install stable-baselines3[extra]")
    print("\n")

    print("--- [4/4] Extracting Data & Formatting Presentation Plots ---")
    try:
        # Run scripts relative to their location
        subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "generate_ml_metrics.py")], check=True)
        subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "extract_paper_metrics.py")], check=True)
        subprocess.run([sys.executable, os.path.join(SCRIPTS_DIR, "generate_presentation_plots.py")], check=True)
        print("[OK] All visual artifacts completely generated in /output.")
    except Exception as e:
        print(f"[Error] Visual generation failed: {e}")
    print("\n")

    print("==================================================")
    print("[SUCCESS] ALL MODELS TRAINED AND SECURED IN /models")
    print("==================================================")

if __name__ == "__main__":
    train_all()
