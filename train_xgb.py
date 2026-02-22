import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import xgboost as xgb

# Import Local Modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from collision.risk_predictor import CollisionRiskModel

def prepare_dataset():
    """Load or synthesize robust training parameters for Risk XGBoost."""
    ds_path = os.path.join(BASE_DIR, "data", "conjuction_and_constellation_data.csv")
    
    if os.path.exists(ds_path):
        df = pd.read_csv(ds_path)
    else:
        df = pd.DataFrame()

    # The parsed simulation data is extremely sparse on exact hits (thankfully for satellites). 
    # For a robust ML model capable of differentiating risk, we synthesize realistic perturbations.
    print(f"📊 Rebalancing dataset with synthesized HIGH/MEDIUM Risk encounters...")
    
    np.random.seed(42)
    # 500 Safe (LOW), 300 Close (MEDIUM), 200 Critical (HIGH)
    low_dist = np.random.uniform(20.0, 100.0, 500)
    low_vel = np.random.uniform(1.0, 15.0, 500)
    
    med_dist = np.random.uniform(5.0, 20.0, 300)
    med_vel = np.random.uniform(8.0, 15.0, 300)
    
    high_dist = np.random.uniform(0.1, 5.0, 200)
    high_vel = np.random.uniform(10.0, 18.0, 200)
    
    df_synth = pd.DataFrame({
        "miss_distance_km": np.concatenate([low_dist, med_dist, high_dist]),
        "relative_velocity_km_s": np.concatenate([low_vel, med_vel, high_vel]),
        "risk_label": ["LOW"]*500 + ["MEDIUM"]*300 + ["HIGH"]*200
    })
    
    df = pd.concat([df, df_synth], ignore_index=True) if not df.empty else df_synth
    return df

def train_and_evaluate_xgboost():
    print("🚀 Initializing XGBoost Collision Risk Training sequence...")
    df = prepare_dataset()
    
    predictor = CollisionRiskModel()
    
    X = df[["miss_distance_km", "relative_velocity_km_s"]].values
    y = predictor.label_encoder.transform(df["risk_label"].values)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"🧠 Training Multi-Class Tree on N={len(X_train)} samples...")
    predictor.model.fit(X_train, y_train)
    
    # 1. Predictions & Accuracy
    preds = predictor.model.predict(X_test)
    probs = predictor.model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Accuracy: {acc:.4f}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    print(f"📊 Confusion Matrix:\n{cm}")
    
    # 3. Save Model Weights
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    model_dst = os.path.join(BASE_DIR, "models", "xgboost_collision.json")
    predictor.model.save_model(model_dst)
    
    # 4. Multi-class ROC Curve Output
    classes = predictor.label_encoder.classes_
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'red']
    for i, color in zip(range(3), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc:0.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Collision Risk')
    plt.legend(loc="lower right")
    
    graph_out = os.path.join(BASE_DIR, "models", "xgboost_roc_curve.png")
    plt.savefig(graph_out)
    print(f"📈 Evaluation Plot exported to {graph_out}")

if __name__ == "__main__":
    train_and_evaluate_xgboost()
