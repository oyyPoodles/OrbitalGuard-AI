import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import torch
import torch.nn as nn
import pandas as pd

# ─── Path Alignment ───────────────────────────────────────
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT = os.path.dirname(SCRIPTS_DIR)
PROJECT_ROOT = os.path.dirname(AIML_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, 'backend')

if BACKEND_ROOT not in sys.path: sys.path.insert(0, BACKEND_ROOT)
if AIML_ROOT not in sys.path: sys.path.insert(0, AIML_ROOT)

from collision.risk_model import RiskClassifier, generate_training_data
from prediction.lstm_model import ResidualTrajectoryLSTM, generate_hybrid_dataset

def plot_xgboost_metrics():
    print("Generating XGBoost Metrics...")
    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    OUTPUT_DIR = os.path.join(AIML_ROOT, 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rc = RiskClassifier(model_path=os.path.join(MODELS_DIR, 'xgb_risk.pkl'))
    X_test, y_test = generate_training_data(1000)
    
    # Predict
    y_pred = rc.model.predict(X_test)
    y_prob = rc.model.predict_proba(X_test)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['LOW', 'MEDIUM', 'HIGH'], 
                yticklabels=['LOW', 'MEDIUM', 'HIGH'])
    plt.title('XGBoost Risk Classifier - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Reliability Diagram (Calibration Curve for HIGH risk class 2)
    prob_true, prob_pred = calibration_curve(y_test == 2, y_prob[:, 2], n_bins=5)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.title('Reliability Diagram (HIGH Risk Calibration)')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of True Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'reliability_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved metrics to {OUTPUT_DIR}")
    
def generate_lstm_training_curve():
    print("Generating LSTM Training Loss Curve...")
    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    OUTPUT_DIR = os.path.join(AIML_ROOT, 'output')
    
    loss_path = os.path.join(MODELS_DIR, 'lstm_loss.npy')
    if not os.path.exists(loss_path):
        print(f"LSTM loss array not found at {loss_path}. Please run train_all_models.py first.")
        return
        
    losses = np.load(loss_path)
    epochs = len(losses)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs+1), losses, linewidth=2, color='green')
    plt.title('LSTM Residual Model - Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved training_curves.png to {OUTPUT_DIR}")

def generate_ppo_reward_curve():
    print("Generating PPO RL Training Rewards Curve...")
    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    OUTPUT_DIR = os.path.join(AIML_ROOT, 'output')
    monitor_path = os.path.join(MODELS_DIR, 'ppo_monitor.monitor.csv')
    
    if not os.path.exists(monitor_path):
        print(f"PPO Monitor logs not found at {monitor_path}.")
        return
        
    df = pd.read_csv(monitor_path, skiprows=1)
    window = max(int(len(df) / 10), 5)
    rolling_reward = df['r'].rolling(window=window).mean()
    
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df['r'], alpha=0.3, color='orange', label='Episodic Reward')
    plt.plot(df.index, rolling_reward, color='red', linewidth=3, label=f'{window}-Episode Moving Avg')
    plt.title('PPO Autonomous Collision Avoidance - Episodic Rewards')
    plt.xlabel('Episodes Trained')
    plt.ylabel('Accumulated Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'ppo_rewards.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved ppo_rewards.png to {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_xgboost_metrics()
    generate_lstm_training_curve()
    generate_ppo_reward_curve()
