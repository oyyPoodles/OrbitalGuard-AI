"""
STEP 3: Hybrid LSTM Trajectory Prediction Model (Residual Corrector)
Input: Sequence of past orbital states (e.g., past 10 time steps of [x, y, z, vx, vy, vz])
Output: Residual correction vector [dx, dy, dz] to add to the deterministic SGP4 prediction
"""
import numpy as np
import torch
import torch.nn as nn
import os

class ResidualTrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, num_layers=2):
        super().__init__()
        # Standard LSTM layer to process temporal tracking data
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Fully connected layer maps hidden state to the 3D spatial residual (dx, dy, dz)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        out, _ = self.lstm(x)
        # We only care about the prediction after seeing the entire sequence
        return self.fc(out[:, -1, :])  


def generate_hybrid_dataset(n_samples=1000, seq_len=10, horizon=1):
    """
    Generate a synthetic dataset representing the academic experiment:
    - SGP4 Deterministic Path: A baseline predictable path
    - Ground Truth: SGP4 path + cumulative non-deterministic drift (e.g., drag/radiation)
    
    The LSTM must learn to map past sequences to the future (SGP4 - Ground Truth) delta.
    """
    X = []
    y = []
    
    # Generate random sequences simulating LEO orbits
    for _ in range(n_samples):
        # Simulated orbital radius (LEO)
        r = 7000 + np.random.uniform(-500, 500)
        t = np.linspace(0, 2 * np.pi, seq_len + horizon)
        
        # SGP4 Deterministic Baseline (pure physics)
        sgp4_x = r * np.cos(t)
        sgp4_y = r * np.sin(t)
        sgp4_z = np.zeros_like(t)
        
        # Generate a predictable but complex drift pattern (simulating atmospheric drag/solar radiation)
        # We use a quadratic trend mixed with a sine wave to represent orbital decay + daily thermal cycles
        decay_factor = np.random.uniform(0.005, 0.015)
        thermal_cycle = np.random.uniform(0.5, 1.5)
        
        drift_x = -decay_factor * (t ** 2) + thermal_cycle * np.sin(t * 2) 
        drift_y = -decay_factor * (t ** 2.1) + thermal_cycle * np.cos(t * 2)
        drift_z = - (decay_factor / 2) * t + 0.1 * np.sin(t)
        
        # Add a tiny bit of white noise to make it realistic 
        drift_x += np.random.normal(0, 0.1, len(t))
        drift_y += np.random.normal(0, 0.1, len(t))
        drift_z += np.random.normal(0, 0.05, len(t))
        
        # Ground Truth = Physics + Non-deterministic Perturbations
        true_x = sgp4_x + drift_x
        true_y = sgp4_y + drift_y
        true_z = sgp4_z + drift_z
        
        # Input features: We feed the network the recent GROUND TRUTH history
        # In reality, this would be Kalman-filtered radar data
        vx = np.gradient(true_x)
        vy = np.gradient(true_y)
        vz = np.gradient(true_z)
        
        sequence = np.stack([true_x, true_y, true_z, vx, vy, vz], axis=-1)
        
        # The target label is the RESIDUAL (Ground Truth - SGP4 Baseline) at time T+horizon
        target_residual_x = true_x[seq_len + horizon - 1] - sgp4_x[seq_len + horizon - 1]
        target_residual_y = true_y[seq_len + horizon - 1] - sgp4_y[seq_len + horizon - 1]
        target_residual_z = true_z[seq_len + horizon - 1] - sgp4_z[seq_len + horizon - 1]
        
        target_residual = np.array([target_residual_x, target_residual_y, target_residual_z])
        
        X.append(sequence[:seq_len])
        y.append(target_residual)

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


def train_hybrid_model(save_path='models/hybrid_lstm.pth', epochs=40, seq_len=10):
    """Train the Residual LSTM on tracking sequences to predict drift."""
    model = ResidualTrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train = generate_hybrid_dataset(n_samples=2000, seq_len=seq_len)

    print("--- Training Hybrid SGP4 + LSTM Residual Model ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred_residual = model(X_train)
        loss = criterion(pred_residual, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Residual MSE Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Hybrid LSTM model saved to {save_path}")
    return model


def load_hybrid_model(path='models/hybrid_lstm.pth'):
    """Load the pre-trained Hybrid LSTM model."""
    model = ResidualTrajectoryLSTM()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def predict_hybrid_correction(model, past_states):
    """
    Predict the (dx, dy, dz) drift correction for the next time step.
    
    Args:
        model: ResidualTrajectoryLSTM instance
        past_states: np.array of shape (seq_len, 6) containing [x,y,z,vx,vy,vz] tracking history
        
    Returns:
        np.array of shape (3,) representing [dx, dy, dz] correction
    """
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(past_states).unsqueeze(0)  # Add batch dim (1, seq_len, 6)
        correction = model(x)
    return correction.squeeze().numpy()


if __name__ == "__main__":
    train_hybrid_model()
