# -*- coding: utf-8 -*-
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
OrbitalGuard AI - Model Unit Tests
====================================
Tests each model component independently with clear pass/fail assertions.

Usage:
  python aiml/scripts/test_models.py
  python aiml/scripts/test_models.py --verbose

Exit code: 0 if all pass, 1 if any fail.
"""

import os
import sys
import traceback
import numpy as np
import argparse

# ─── Path Alignment ──────────────────────────────────────
SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT    = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(AIML_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, 'backend')

for p in [AIML_ROOT, BACKEND_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

MODELS_DIR = os.path.join(AIML_ROOT, 'models')
DATA_DIR   = os.path.join(AIML_ROOT, 'data')

# ────────────────────────────────────────────────────────
PASS = "[PASS]"
FAIL = "[FAIL]"


def run_test(name: str, fn, verbose: bool = False) -> bool:
    """Run a single test function, catch exceptions, print result."""
    print(f"\n  Testing: {name} ...", end=" ")
    try:
        fn()
        print(PASS)
        return True
    except AssertionError as e:
        print(f"{FAIL}  — {e}")
        if verbose:
            traceback.print_exc()
        return False
    except Exception as e:
        print(f"{FAIL}  — Unexpected error: {e}")
        if verbose:
            traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════
# TEST 1: Debris Catalog
# ════════════════════════════════════════════════════════
def test_debris_catalog():
    from knowledge.debris_catalog import DebrisCatalog

    catalog = DebrisCatalog()
    stats   = catalog.get_stats()
    assert stats['total'] > 0, "Catalog is empty"

    # ID lookup
    first = catalog._entries[0]
    found = catalog.get_by_id(first['id'])
    assert found is not None, f"Failed to find first entry by ID: {first['id']}"
    assert found['id'] == first['id']

    # Name search
    results = catalog.search_by_name(first['name'][:5])
    assert len(results) > 0, "Name search returned no results"

    # Describe object
    desc = catalog.describe_object(found)
    assert len(desc) > 10, "describe_object returned too short a string"


# ════════════════════════════════════════════════════════
# TEST 2: NaradChatbot
# ════════════════════════════════════════════════════════
def test_chatbot():
    from knowledge.orbital_chatbot import NaradChatbot

    bot = NaradChatbot()
    assert bot.ready, "Chatbot not ready — check debt_catalog.json path"

    # NORAD ID lookup
    resp = bot.ask("What is 25544?")
    assert "25544" in resp or "ZARYA" in resp.upper() or "NORAD" in resp, \
        f"ID lookup response unexpected: {resp[:100]}"

    # Stats query
    resp_stats = bot.ask("How many objects are tracked?")
    assert "Total" in resp_stats or "tracked" in resp_stats.lower(), \
        f"Stats response unexpected: {resp_stats[:100]}"

    # Type query
    resp_type = bot.ask("What are debris objects?")
    assert len(resp_type) > 20, "Type response too short"

    # Help query
    resp_help = bot.ask("Help")
    assert "Narad" in resp_help or "can answer" in resp_help.lower()


# ════════════════════════════════════════════════════════
# TEST 3: TLE Preprocessor (fast mode with 50 objects)
# ════════════════════════════════════════════════════════
def test_preprocessor():
    from preprocessing.tle_preprocessor import TLEPreprocessor

    preproc = TLEPreprocessor(max_objects=50, n_propagation_steps=20)
    paths   = preproc.run()

    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        arr = np.load(paths[key])
        assert arr.ndim == (3 if 'X' in key else 2), \
            f"Wrong ndim for {key}: {arr.ndim}"
        assert arr.shape[0] > 0, f"Empty array for {key}"

    # Check shapes consistent
    X_tr = np.load(paths['X_train'])
    y_tr = np.load(paths['y_train'])
    assert X_tr.shape[0] == y_tr.shape[0], "X/y sample count mismatch"
    assert X_tr.shape[-1] == 6, f"Expected 6 features, got {X_tr.shape[-1]}"
    assert y_tr.shape[-1] == 3, f"Expected 3-dim target, got {y_tr.shape[-1]}"


# ════════════════════════════════════════════════════════
# TEST 4: LSTM Model
# ════════════════════════════════════════════════════════
def test_lstm():
    import torch
    from prediction.lstm_model import (
        ResidualTrajectoryLSTM,
        generate_hybrid_dataset,
        predict_hybrid_correction
    )

    # Instantiation
    model = ResidualTrajectoryLSTM(input_dim=6, hidden_dim=64, output_dim=3)
    assert sum(p.numel() for p in model.parameters()) > 0, "Model has no parameters"

    # Forward pass
    X, y = generate_hybrid_dataset(n_samples=32, seq_len=10)
    model.eval()
    with torch.no_grad():
        out = model(X)
    assert out.shape == (32, 3), f"Expected (32,3), got {out.shape}"

    # Single prediction
    past = np.random.randn(10, 6).astype(np.float32)
    correction = predict_hybrid_correction(model, past)
    assert correction.shape == (3,), f"Expected (3,), got {correction.shape}"
    assert not np.any(np.isnan(correction)), "NaN in prediction"

    # Load from disk if available
    lstm_path = os.path.join(MODELS_DIR, 'hybrid_lstm.pth')
    if os.path.exists(lstm_path):
        from prediction.lstm_model import load_hybrid_model, evaluate
        loaded = load_hybrid_model(lstm_path)
        loaded.eval()

        # Check output on dummy input
        with torch.no_grad():
            out2 = loaded(X[:4])
        assert out2.shape == (4, 3)


# ════════════════════════════════════════════════════════
# TEST 5: XGBoost Risk Classifier
# ════════════════════════════════════════════════════════
def test_xgboost():
    from collision.risk_model import RiskClassifier

    xgb_path = os.path.join(MODELS_DIR, 'xgb_risk.pkl')
    rc       = RiskClassifier(model_path=xgb_path, use_real_data=False)

    # Sanity classification
    assert rc.classify(0.5, 8.0)   == "HIGH",   "Expected HIGH for close approach"
    assert rc.classify(20.0, 2.0)  == "LOW",    "Expected LOW for distant object"

    # Confidence scores
    result = rc.classify_with_confidence(0.8, 9.0)
    assert 'risk_level'    in result
    assert 'confidence'    in result
    assert 'probabilities' in result
    assert abs(sum(result['probabilities'].values()) - 1.0) < 0.01, \
        "Probabilities should sum to ~1.0"

    # Batch classification
    conjunctions = [
        {'distance_km': 0.5,  'relative_velocity_kms': 8.0},
        {'distance_km': 20.0, 'relative_velocity_kms': 1.0},
    ]
    result_batch = rc.classify_batch(conjunctions)
    assert result_batch[0]['risk_level'] == 'HIGH'
    assert result_batch[1]['risk_level'] == 'LOW'


# ════════════════════════════════════════════════════════
# TEST 6: PPO Agent (rule-based fallback — no SB3 needed)
# ════════════════════════════════════════════════════════
def test_ppo():
    from avoidance.ppo_agent import PPOAvoidanceAgent, CollisionAvoidanceEnv

    # Environment smoke test
    env = CollisionAvoidanceEnv(curriculum_phase=1)
    obs, info = env.reset(seed=42)
    assert obs.shape == (12,), f"Expected obs shape (12,), got {obs.shape}"

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2.shape == (12,)
    assert isinstance(reward, float)

    # Agent avoidance output
    ppo_path = os.path.join(MODELS_DIR, 'ppo_avoidance.zip')
    agent    = PPOAvoidanceAgent(model_path=ppo_path)

    rel_pos = np.array([5.0, 0.0, 0.0], dtype=np.float32)
    rel_vel = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    own_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    own_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    result = agent.compute_avoidance(rel_pos, rel_vel, own_pos, own_vel)
    assert 'delta_v'   in result
    assert 'fuel_cost' in result
    assert result['delta_v'].shape == (3,)
    assert result['fuel_cost'] >= 0.0


# ════════════════════════════════════════════════════════
# TEST 7: Collision Detector (KDTree)
# ════════════════════════════════════════════════════════
def test_collision_detector():
    from collision.detector import CollisionDetector

    detector = CollisionDetector(threshold_km=5.0)

    objects = [
        {'id': '1', 'name': 'SAT-A', 'position': np.array([0.0, 0.0, 0.0]), 'velocity': np.array([1.0, 0.0, 0.0]), 'type': 'payload'},
        {'id': '2', 'name': 'SAT-B', 'position': np.array([3.0, 0.0, 0.0]), 'velocity': np.array([-1.0, 0.0, 0.0]), 'type': 'debris'},   # 3 km apart → should detect
        {'id': '3', 'name': 'SAT-C', 'position': np.array([100.0, 0.0, 0.0]), 'velocity': np.array([0.0, 1.0, 0.0]), 'type': 'payload'},   # far → no
    ]

    conjunctions = detector.detect(objects)
    assert len(conjunctions) == 1, f"Expected 1 conjunction, got {len(conjunctions)}"
    assert conjunctions[0]['distance_km'] < 5.0


# ════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='OrbitalGuard AI — Model Tests')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ORBITALGUARD AI - MODEL TEST SUITE")
    print("=" * 60)

    tests = [
        ("Debris Catalog",       test_debris_catalog),
        ("Narad Chatbot",        test_chatbot),
        ("TLE Preprocessor",     test_preprocessor),
        ("LSTM Model",           test_lstm),
        ("XGBoost Classifier",   test_xgboost),
        ("PPO Agent",            test_ppo),
        ("Collision Detector",   test_collision_detector),
    ]

    results = []
    for name, fn in tests:
        ok = run_test(name, fn, verbose=args.verbose)
        results.append((name, ok))

    # Summary
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed

    print("\n" + "─" * 60)
    print(f"  Results: {passed}/{len(results)} passed")
    if failed:
        print(f"\n  Failed tests:")
        for name, ok in results:
            if not ok:
                print(f"    ✗ {name}")
    print("─" * 60 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
