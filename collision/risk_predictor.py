import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class CollisionRiskModel:
    """
    XGBoost Classifier to predict the severity of a conjunction event.
    Maps:
     Features -> Miss Distance, Relative Velocity
     Labels   -> LOW, MEDIUM, HIGH Risk
    """
    def __init__(self, model_path="models/xgboost_collision.json"):
        self.model_path = model_path
        self.model = XGBClassifier(
            objective="multi:softmax",
            eval_metric="mlogloss",
            num_class=3
        )
        self.label_encoder = LabelEncoder()
        # Ensure consistent label mapping
        self.label_encoder.fit(["LOW", "MEDIUM", "HIGH"])
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[Collision] Trained XGBoost weights missing at {self.model_path}. Run train_xgb.py first.")
            
        self.model.load_model(self.model_path)
        print(f"[Collision] Loaded trained Research-Grade XGBoost model from {self.model_path}")
                
    def predict_risk(self, miss_dist: float, rel_vel: float) -> dict:
        """
        Infers the risk class and pseudo-probability using strict XGBoost.
        """
        features = np.array([[miss_dist, rel_vel]])
        
        # In XGBoost, predict() returns class index
        pred_idx = self.model.predict(features)[0]
        # predict_proba returns array of probabilities for all classes
        probas = self.model.predict_proba(features)[0]
        
        label = self.label_encoder.inverse_transform([pred_idx])[0]
        confidence = probas[pred_idx]
        
        return {
            "class": label,
            "score": float(confidence)
        }

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ds_path = os.path.join(BASE_DIR, "data", "conjuction_and_constellation_data.csv")
    
    predictor = CollisionRiskModel()
    predictor.train_or_mock(ds_path)
    
    sample = predictor.predict_risk(miss_dist=2.5, rel_vel=12.4)
    print(f"\nInference Test (Dist: 2.5km, Vel: 12.4km/s) -> Risk: {sample['class']} ({sample['score']:.2%})")
