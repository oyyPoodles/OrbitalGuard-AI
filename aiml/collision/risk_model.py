"""
STEP 5: XGBoost Collision Risk Classification — Upgraded
=========================================================
Input : distance (km), relative_velocity (km/s),
        angle_between_velocities (deg), altitude_difference (km), mean_altitude (km)
Output: LOW / MEDIUM / HIGH

Improvements over v1:
  - Uses real conjunction CSV data (conjuction_and_constellation_data.csv)
  - 5-feature input (vs 2 previously)
  - 5-fold Stratified cross-validation
  - evaluate(): accuracy, F1 (weighted), ROC-AUC (one-vs-rest)
  - Feature importance plot
  - Probability scores for confidence display
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier

# ─── Path Alignment ──────────────────────────────────────
COLLISION_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT     = os.path.dirname(COLLISION_DIR)
PROJECT_ROOT  = os.path.dirname(AIML_ROOT)
BACKEND_ROOT  = os.path.join(PROJECT_ROOT, 'backend')

CONJUNCTION_CSV = os.path.join(BACKEND_ROOT, 'data', 'conjuction_and_constellation_data.csv')

LABEL_MAP   = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
LABEL_IDX   = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
FEATURE_COLS = [
    'miss_distance_km',
    'relative_velocity_km_s',
    'angle_between_velocities_deg',
    'altitude_difference_km',
    'mean_altitude_km',
]


# ────────────────────────────────────────────────────────
def load_real_conjunction_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and return features + labels from the real conjunction CSV.
    Falls back to synthetic data if CSV is missing or too small.
    """
    if os.path.exists(CONJUNCTION_CSV):
        df = pd.read_csv(CONJUNCTION_CSV).dropna()
        available = [c for c in FEATURE_COLS if c in df.columns]
        if 'risk_label' in df.columns and len(df) >= 5:
            X = df[available].values.astype(np.float32)
            y = df['risk_label'].map(LABEL_IDX).fillna(0).values.astype(int)
            print(f"[XGBoost] Loaded {len(df)} real conjunction records")
            return X, y

    print("[XGBoost] Real data insufficient — using synthetic training data")
    return generate_training_data(n=3000)


def generate_training_data(n: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic labeled conjunction data with 5 features.
    Statistically balanced across LOW / MEDIUM / HIGH classes.
    """
    X = np.zeros((n, 5), dtype=np.float32)
    y = np.zeros(n, dtype=int)

    for i in range(n):
        rv = np.random.rand()
        if rv < 0.33:
            dist = np.random.uniform(0.1, 1.2)
        elif rv < 0.66:
            dist = np.random.uniform(0.8, 5.2)
        else:
            dist = np.random.uniform(4.8, 20.0)

        vel   = np.random.uniform(0.5, 15.0)
        angle = np.random.uniform(0, 180)
        alt_d = np.random.uniform(0, 500)
        alt   = np.random.uniform(300, 2000)

        X[i] = [dist, vel, angle, alt_d, alt]

        eff_dist = dist + np.random.normal(0, 0.25)
        if eff_dist < 1.0:
            y[i] = 2   # HIGH
        elif eff_dist < 5.0:
            y[i] = 1   # MEDIUM
        else:
            y[i] = 0   # LOW

    return X, y


# ────────────────────────────────────────────────────────
class RiskClassifier:
    """
    XGBoost-based collision risk classifier with evaluation + cross-validation.
    """

    def __init__(self, model_path: str = 'models/xgb_risk.pkl',
                 use_real_data: bool = True):
        self.model_path    = model_path
        self.use_real_data = use_real_data
        self.model: XGBClassifier = None
        self.n_features    = len(FEATURE_COLS)
        self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"[XGBoost] Loaded from {self.model_path}")
        else:
            self.train_and_save()

    def train_and_save(self, cv_folds: int = 5):
        """Train XGBoost with optional cross-validation, save model."""
        print("[XGBoost] Training Risk Classifier...")

        if self.use_real_data:
            X, y = load_real_conjunction_data()
        else:
            X, y = generate_training_data(3000)

        # Augment small real datasets with synthetic
        if len(X) < 200:
            X_syn, y_syn = generate_training_data(2000)
            X = np.vstack([X, X_syn[:, :X.shape[1]]])
            y = np.concatenate([y, y_syn])
            print(f"[XGBoost] Augmented with synthetic data → {len(X)} total samples")

        # ── Cross-validation ────────────────────────────────
        if cv_folds > 1 and len(X) >= cv_folds * 3:
            skf    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_acc = []
            for fold, (tr, te) in enumerate(skf.split(X, y)):
                clf = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric='mlogloss', use_label_encoder=False, random_state=42
                )
                clf.fit(X[tr], y[tr])
                preds = clf.predict(X[te])
                acc   = accuracy_score(y[te], preds)
                cv_acc.append(acc)
                print(f"  Fold {fold+1}/{cv_folds} accuracy: {acc:.4f}")

            print(f"  Mean CV Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

        # ── Final model on all data ─────────────────────────
        self.model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', use_label_encoder=False, random_state=42
        )
        self.model.fit(X, y)

        os.makedirs(os.path.dirname(self.model_path) or '.', exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[OK] XGBoost model saved → {self.model_path}")

    def evaluate(self, data_dir: str = None) -> dict:
        """
        Evaluate on held-out test data.
        Returns accuracy, F1 (weighted), ROC-AUC (OvR macro).
        """
        X, y = generate_training_data(1000)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                    stratify=y, random_state=42)

        preds     = self.model.predict(X_te)
        probs     = self.model.predict_proba(X_te)
        acc       = accuracy_score(y_te, preds)
        f1        = f1_score(y_te, preds, average='weighted', zero_division=0)
        y_bin     = label_binarize(y_te, classes=[0, 1, 2])
        try:
            auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')

        print("\n[XGBoost Evaluation]")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}  (weighted)")
        print(f"  ROC-AUC  : {auc:.4f}  (OvR macro)")
        print("\n" + classification_report(y_te, preds,
                                          target_names=['LOW', 'MEDIUM', 'HIGH'],
                                          zero_division=0))
        return {'accuracy': acc, 'f1_weighted': f1, 'roc_auc': auc}

    def classify(self, distance_km: float,
                 relative_velocity_kms: float,
                 angle_deg: float = 90.0,
                 alt_diff_km: float = 50.0,
                 mean_alt_km: float = 550.0) -> str:
        """
        Classify collision risk level for a single conjunction.

        Returns: "LOW", "MEDIUM", or "HIGH"
        """
        feats = self._build_features(distance_km, relative_velocity_kms,
                                     angle_deg, alt_diff_km, mean_alt_km)
        pred = self.model.predict(feats)[0]
        return LABEL_MAP.get(int(pred), "UNKNOWN")

    def classify_with_confidence(self, distance_km: float,
                                  relative_velocity_kms: float,
                                  angle_deg: float = 90.0,
                                  alt_diff_km: float = 50.0,
                                  mean_alt_km: float = 550.0) -> dict:
        """
        Classify and return probability scores for each risk level.
        Used by RightAnalyticsPanel for confidence score display.

        Returns dict:
          {
            'risk_level': 'HIGH',
            'confidence': 0.92,
            'probabilities': {'LOW': 0.02, 'MEDIUM': 0.06, 'HIGH': 0.92}
          }
        """
        feats = self._build_features(distance_km, relative_velocity_kms,
                                     angle_deg, alt_diff_km, mean_alt_km)
        probs = self.model.predict_proba(feats)[0]
        pred  = int(self.model.predict(feats)[0])
        label = LABEL_MAP.get(pred, "UNKNOWN")

        return {
            'risk_level': label,
            'confidence': round(float(probs[pred]), 4),
            'probabilities': {
                'LOW':    round(float(probs[0]), 4),
                'MEDIUM': round(float(probs[1]), 4),
                'HIGH':   round(float(probs[2]), 4),
            }
        }

    def classify_batch(self, conjunctions: List[Dict]) -> List[Dict]:
        """
        Add risk_level + confidence to each conjunction dict.

        Args:
            conjunctions: list of dicts with at minimum:
                'distance_km', 'relative_velocity_kms'
        Returns:
            Same list with 'risk_level', 'confidence', 'probabilities' added.
        """
        for c in conjunctions:
            result = self.classify_with_confidence(
                distance_km=c.get('distance_km', 10.0),
                relative_velocity_kms=c.get('relative_velocity_kms', 1.0),
                angle_deg=c.get('angle_deg', 90.0),
                alt_diff_km=c.get('alt_diff_km', 50.0),
                mean_alt_km=c.get('mean_alt_km', 550.0),
            )
            c['risk_level']    = result['risk_level']
            c['confidence']    = result['confidence']
            c['probabilities'] = result['probabilities']
        return conjunctions

    def feature_importance(self, save_path: str = None) -> Dict[str, float]:
        """Return feature importance scores (and optionally save a plot)."""
        scores = self.model.feature_importances_
        names  = FEATURE_COLS[:len(scores)]
        importance = dict(zip(names, scores.tolist()))

        if save_path:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.barh(list(importance.keys()), list(importance.values()),
                        color='#22d3ee')
                ax.set_title('XGBoost Feature Importance — Risk Classifier')
                ax.set_xlabel('Importance Score')
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                print(f"[OK] Feature importance saved → {save_path}")
            except ImportError:
                print("[Warning] matplotlib not installed; skipping plot")

        print("\n[XGBoost Feature Importance]")
        for name, score in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"  {name:<40} {score:.4f}")
        return importance

    def _build_features(self, distance_km, velocity_kms,
                        angle_deg, alt_diff_km, mean_alt_km) -> np.ndarray:
        """Build a (1, N_features) feature array, padding if needed."""
        full = np.array([[distance_km, velocity_kms, angle_deg,
                          alt_diff_km, mean_alt_km]], dtype=np.float32)
        n_model = len(self.model.feature_importances_)
        return full[:, :n_model]


# ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Force retrain')
    args = parser.parse_args()

    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'xgb_risk.pkl')

    if args.retrain and os.path.exists(model_path):
        os.remove(model_path)

    rc = RiskClassifier(model_path=model_path, use_real_data=True)
    rc.evaluate()
    rc.feature_importance(save_path=os.path.join(AIML_ROOT, 'output', 'feature_importance.png'))

    # Quick sanity checks
    print("\n[Sanity Checks]")
    print(f"  (0.5 km, 8 km/s)   → {rc.classify(0.5, 8.0)}")
    print(f"  (3.0 km, 5 km/s)   → {rc.classify(3.0, 5.0)}")
    print(f"  (20.0 km, 2 km/s)  → {rc.classify(20.0, 2.0)}")

    res = rc.classify_with_confidence(0.8, 9.0)
    print(f"\n  Confidence test: {res}")
