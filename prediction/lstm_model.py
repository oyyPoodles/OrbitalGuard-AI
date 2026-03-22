"""
STEP 3: LSTM Trajectory Prediction Model
Input: past trajectory sequence [x, y, z, vx, vy, vz]
Output: predicted future position [x, y, z]
"""
import numpy as np
import torch
import torch.nn as nn
import os

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, 6)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # last time step → predicted position


def generate_synthetic_data(n_samples=500, seq_len=10):
    """Generate synthetic orbital trajectory sequences for training."""
    X = []
    y = []
    for _ in range(n_samples):
        # Simulate a simple circular-ish orbit with noise
        t = np.linspace(0, 2 * np.pi, seq_len + 1)
        r = 7000 + np.random.uniform(-500, 500)
        x = r * np.cos(t) + np.random.normal(0, 10, len(t))
        yc = r * np.sin(t) + np.random.normal(0, 10, len(t))
        z = np.random.normal(0, 100, len(t))
        vx = -r * np.sin(t) * 0.001
        vy = r * np.cos(t) * 0.001
        vz = np.random.normal(0, 0.01, len(t))

        seq = np.stack([x, yc, z, vx, vy, vz], axis=-1)
        X.append(seq[:seq_len])
        y.append(seq[seq_len, :3])  # predict next position

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


def train_model(save_path='models/lstm.pth', epochs=20):
    """Train the LSTM on synthetic orbital data."""
    model = TrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_train, y_train = generate_synthetic_data(500, 10)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ LSTM model saved to {save_path}")
    return model


def load_model(path='models/lstm.pth'):
    """Load a pre-trained LSTM model."""
    model = TrajectoryLSTM()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def predict_trajectory(model, past_states):
    """
    Predict next position from a sequence of past states.
    
    Args:
        model: TrajectoryLSTM instance
        past_states: np.array of shape (seq_len, 6)
        
    Returns:
        predicted position np.array of shape (3,)
    """
    with torch.no_grad():
        x = torch.FloatTensor(past_states).unsqueeze(0)  # (1, seq_len, 6)
        pred = model(x)
    return pred.squeeze().numpy()


if __name__ == "__main__":
    train_model()
