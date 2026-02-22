import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# Set path for sibling modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from prediction.lstm_predictor import TrajectoryLSTM
from utils.data_processor import DataProcessor
from simulation.propagator import OrbitalPropagator

def create_sequences(data, seq_length=10):
    """
    Given a massive array of time-series trajectories for a satellite,
    creates rolling windows (X) and next-step targets (y).
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def generate_training_data():
    """Uses the SGP4 Propagator as physics ground-truth for LSTM label generation."""
    print("🌍 Extracting SGP4 Ground-Truth Trajectories...")
    
    dp = DataProcessor(
        os.path.join(BASE_DIR, "data", "tle_data.txt"),
        os.path.join(BASE_DIR, "data", "conjuction_and_constellation_data.csv")
    )
    sats = dp.load_tles()
    
    # We will train on a sample of 100 satellites over 120 minutes
    # to maintain local processing overhead
    if len(sats) > 100: sats = sats[:100]
    
    propagator = OrbitalPropagator(sats)
    trajectories = propagator.simulate_trajectory(datetime.utcnow(), duration_mins=120, step_size_mins=1)
    
    all_X, all_y = [], []
    for sat_id, data in trajectories.items():
        pos_array = np.array(data['positions']) # Shape (120, 3)
        X, y = create_sequences(pos_array, seq_length=10)
        all_X.append(X)
        all_y.append(y)
        
    final_X = np.vstack(all_X)
    final_y = np.vstack(all_y)
    print(f"✅ Generated {len(final_X)} training sequences.")
    
    # Simple standardization
    mean = np.mean(final_X, axis=(0, 1))
    std = np.std(final_X, axis=(0, 1)) + 1e-8
    
    final_X = (final_X - mean) / std
    final_y = (final_y - mean) / std
    
    return final_X, final_y, mean, std

def train_lstm_model():
    X, y, mean, std = generate_training_data()
    
    # Convert to PyTorch tensors
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 Initializing LSTM on device: {device}")
    
    model = TrajectoryLSTM(input_dim=3, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
            
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_outputs = model(batch_x)
                v_loss = criterion(val_outputs, batch_y)
                val_batch_losses.append(v_loss.item())
                
        val_loss = np.mean(val_batch_losses)
        val_losses.append(val_loss)
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss (MSE): {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Calculate final RMSE on raw unscaled terms
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x.to(device)).cpu().numpy()
            all_preds.append(outputs * std + mean)
            all_truths.append(batch_y.numpy() * std + mean)
            
    all_preds = np.vstack(all_preds)
    all_truths = np.vstack(all_truths)
    
    rmse = np.sqrt(np.mean((all_preds - all_truths)**2))
    print(f"\n🎯 Final Validation RMSE: {rmse:.2f} km")
    
    # Save Model
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, "lstm_trajectory.pth")
    torch.save(model.state_dict(), save_path)
    print(f"📁 Trained Model Checkpoint saved to: {save_path}")
    
    # Plot curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('LSTM Trajectory Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(model_dir, "lstm_loss_curve.png")
    plt.savefig(plot_path)
    print(f"📈 Loss curve saved to: {plot_path}")

if __name__ == "__main__":
    train_lstm_model()
