import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import the Hybrid model we just built
from lstm_model import ResidualTrajectoryLSTM, generate_hybrid_dataset, train_hybrid_model

def calculate_rmse(predictions, ground_truths):
    """Calculate the Root Mean Square Error between predicted and actual coordinates."""
    return np.sqrt(np.mean(np.sum((predictions - ground_truths)**2, axis=1)))

def run_experiment(horizons=[1, 5, 10]):
    """
    Run the core academic experiment comparing Pure SGP4 vs Hybrid SGP4+LSTM.
    
    1. Generate an evaluation dataset with realistic deterministic + non-deterministic drift.
    2. Establish the Pure SGP4 baseline error.
    3. Train the Hybrid LSTM to predict the residuals.
    4. Calculate the Hybrid SGP4+LSTM error.
    5. Print quantifiable metrics and generate experimental plots.
    """
    print("==================================================")
    print("[RESEARCH EXPERIMENT] Pure SGP4 vs Hybrid SGP4+LSTM")
    print("==================================================\n")

    # 1. Train the Hybrid Model
    print("[1] Training Hybrid Model on historical sequence data...")
    model_path = 'models/hybrid_lstm.pth'
    if not os.path.exists(model_path):
        model = train_hybrid_model(save_path=model_path, epochs=30, seq_len=10)
    else:
        model = ResidualTrajectoryLSTM()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"[OK] Loaded existing model from {model_path}")
    model.eval()

    # Data structures for plotting
    sgp4_rmse_list = []
    hybrid_rmse_list = []
    improvements = []

    # 2. Run Horizons Evaluation
    for horizon in horizons:
        print(f"\n--- Evaluating at Prediction Horizon: T+{horizon} steps ---")
        
        # We generate a testing loop internally simulating the drift
        X_test, y_test_residuals = generate_hybrid_dataset(n_samples=200, seq_len=10, horizon=horizon)
        
        # Ground Truth Residuals (The actual physical drift accumulated at T+horizon)
        y_test_residuals = y_test_residuals.numpy() 

        # Pure SGP4 inherently predicts 0 drift (deterministic), so its error is simply the magnitude of the actual drift
        # because SGP4_Predicted = SGP4_Baseline. So error = ||SGP4_Predicted - Ground_Truth|| = ||0 - Residual||
        sgp4_rmse = np.sqrt(np.mean(np.sum((0 - y_test_residuals)**2, axis=1)))
        
        # Hybrid Model Prediction
        with torch.no_grad():
            predicted_residuals = model(X_test).numpy()
            
        # Hybrid Error = ||Predicted_Residual - Actual_Residual||
        hybrid_rmse = np.sqrt(np.mean(np.sum((predicted_residuals - y_test_residuals)**2, axis=1)))
        
        improvement = ((sgp4_rmse - hybrid_rmse) / sgp4_rmse) * 100
        
        sgp4_rmse_list.append(sgp4_rmse)
        hybrid_rmse_list.append(hybrid_rmse)
        improvements.append(improvement)

        print(f"  • Pure SGP4 RMSE:    {sgp4_rmse:.2f} km")
        print(f"  • Hybrid Model RMSE: {hybrid_rmse:.2f} km")
        print(f"  • Error Reduction:   {improvement:.2f}%")

    print("\n==================================================")
    avg_improvement = np.mean(improvements)
    print(f"[EXPERIMENT CONCLUDED] The Hybrid SGP4+LSTM architecture")
    print(f"   reduced positional prediction error by an average of {avg_improvement:.1f}%.")
    print("==================================================")

    # 3. Generate Plot
    plot_path = "models/experiment_results.png"
    plt.figure(figsize=(10, 6))
    plt.plot(horizons, sgp4_rmse_list, marker='o', label='Pure SGP4 (Deterministic)', linewidth=2, color='red', linestyle='--')
    plt.plot(horizons, hybrid_rmse_list, marker='s', label='Hybrid SGP4 + LSTM', linewidth=3, color='blue')
    
    plt.title('Prediction Accuracy: Pure physics vs. Hybrid AI', fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Horizon (Time Steps)', fontsize=12)
    plt.ylabel('Root Mean Square Error (RMSE km)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[Visualization saved to] {plot_path}")

if __name__ == "__main__":
    run_experiment([1, 2, 5, 10, 15])
