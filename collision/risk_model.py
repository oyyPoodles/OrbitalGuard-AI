"""
STEP 5: XGBoost Collision Risk Classification
Input: distance (km), relative velocity (km/s)
Output: LOW / MEDIUM / HIGH
"""
import numpy as np
import os
import pickle
from xgboost import XGBClassifier

LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def generate_training_data(n=2000):
    """Generate synthetic labeled collision risk data."""
    X = np.zeros((n, 2))
    y = np.zeros(n, dtype=int)

    for i in range(n):
        dist = np.random.uniform(0.1, 50.0)
        vel = np.random.uniform(0.5, 15.0)
        X[i] = [dist, vel]

        if dist < 1.0:
            y[i] = 2  # HIGH
        elif dist < 5.0:
            y[i] = 1  # MEDIUM
        else:
            y[i] = 0  # LOW

    return X, y


class RiskClassifier:
    def __init__(self, model_path='models/xgb_risk.pkl'):
        self.model_path = model_path
        self.model = None
        self._load_or_train()

    def _load_or_train(self):
        """Load existing model or train a new one."""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.train_and_save()

    def train_and_save(self):
        """Train XGBoost on synthetic data and save."""
        print("Training XGBoost Risk Classifier...")
        X, y = generate_training_data(2000)
        self.model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        self.model.fit(X, y)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ XGBoost model saved to {self.model_path}")

    def classify(self, distance_km, relative_velocity_kms):
        """
        Classify collision risk for a single conjunction.
        
        Returns:
            str: "LOW", "MEDIUM", or "HIGH"
        """
        features = np.array([[distance_km, relative_velocity_kms]])
        pred = self.model.predict(features)[0]
        return LABEL_MAP.get(int(pred), "UNKNOWN")

    def classify_batch(self, conjunctions):
        """
        Add risk_level to each conjunction dict.
        
        Args:
            conjunctions: list of dicts with 'distance_km' and 'relative_velocity_kms'
            
        Returns:
            Same list with 'risk_level' added.
        """
        for c in conjunctions:
            c['risk_level'] = self.classify(c['distance_km'], c['relative_velocity_kms'])
        return conjunctions


if __name__ == "__main__":
    rc = RiskClassifier()
    print(rc.classify(0.5, 8.0))   # HIGH
    print(rc.classify(3.0, 5.0))   # MEDIUM
    print(rc.classify(20.0, 2.0))  # LOW
