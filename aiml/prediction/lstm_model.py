"""
STEP 3: Hybrid LSTM Trajectory Prediction Model (Residual Corrector) — Upgraded
================================================================================
Architecture: 2-layer LSTM → FC → residual vector (dx, dy, dz)
Key equation: x_final = x_SGP4 + x_LSTM

Improvements over v1:
  - train_on_real_data(): ingests preprocessed .npy sequences
  - Validation loss tracking + early stopping (patience=5)
  - evaluate(): RMSE on test split
  - fine_tune(): freeze LSTM, retrain FC head only
  - Improved training loop with LR scheduling
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─── Path Alignment ──────────────────────────────────────
PREDICTION_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT      = os.path.dirname(PREDICTION_DIR)
if AIML_ROOT not in sys.path:
    sys.path.insert(0, AIML_ROOT)


# ────────────────────────────────────────────────────────
class ResidualTrajectoryLSTM(nn.Module):
    """
    Hybrid SGP4 + LSTM residual corrector.
    Input  : (batch, seq_len, 6)  — [x,y,z,vx,vy,vz] history
    Output : (batch, 3)           — residual [dx, dy, dz]
    """
    def __init__(self, input_dim: int = 6,
                 hidden_dim: int = 128,
                 output_dim: int = 3,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep → residual


# ────────────────────────────────────────────────────────
def generate_hybrid_dataset(n_samples: int = 1000,
                             seq_len: int = 10,
                             horizon: int = 1):
    """
    Generate synthetic training data (used when real data is unavailable).
    Simulates LEO orbits with deterministic SGP4 + stochastic drift.
    """
    X, y = [], []
    for _ in range(n_samples):
        r = 7000 + np.random.uniform(-500, 500)
        t = np.linspace(0, 2 * np.pi, seq_len + horizon)

        sgp4_x = r * np.cos(t)
        sgp4_y = r * np.sin(t)
        sgp4_z = np.zeros_like(t)

        decay         = np.random.uniform(0.005, 0.015)
        thermal_cycle = np.random.uniform(0.5, 1.5)

        drift_x = -decay * (t ** 2)   + thermal_cycle * np.sin(t * 2)
        drift_y = -decay * (t ** 2.1) + thermal_cycle * np.cos(t * 2)
        drift_z = -(decay / 2) * t    + 0.1 * np.sin(t)

        drift_x += np.random.normal(0, 0.1, len(t))
        drift_y += np.random.normal(0, 0.1, len(t))
        drift_z += np.random.normal(0, 0.05, len(t))

        true_x = sgp4_x + drift_x
        true_y = sgp4_y + drift_y
        true_z = sgp4_z + drift_z

        vx = np.gradient(true_x)
        vy = np.gradient(true_y)
        vz = np.gradient(true_z)

        sequence = np.stack([true_x, true_y, true_z, vx, vy, vz], axis=-1)

        target = np.array([
            true_x[seq_len + horizon - 1] - sgp4_x[seq_len + horizon - 1],
            true_y[seq_len + horizon - 1] - sgp4_y[seq_len + horizon - 1],
            true_z[seq_len + horizon - 1] - sgp4_z[seq_len + horizon - 1],
        ])

        X.append(sequence[:seq_len])
        y.append(target)

    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))


# ────────────────────────────────────────────────────────
def train_hybrid_model(save_path: str = 'models/hybrid_lstm.pth',
                       epochs: int = 40,
                       seq_len: int = 10) -> 'ResidualTrajectoryLSTM':
    """Train on synthetic data (original pipeline, kept for compatibility)."""
    model     = ResidualTrajectoryLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train, y_train = generate_hybrid_dataset(n_samples=2000, seq_len=seq_len)

    print("--- Training Hybrid SGP4 + LSTM Residual Model (Synthetic) ---")
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.save(os.path.join(os.path.dirname(save_path), 'lstm_loss.npy'), np.array(losses))
    torch.save(model.state_dict(), save_path)
    print(f"[OK] Synthetic LSTM saved → {save_path}")
    return model


# ────────────────────────────────────────────────────────
def train_on_real_data(data_dir:    str = None,
                       save_path:   str = 'models/hybrid_lstm.pth',
                       epochs:      int = 60,
                       batch_size:  int = 256,
                       lr:          float = 1e-3,
                       patience:    int = 8) -> 'ResidualTrajectoryLSTM':
    """
    Train LSTM on real preprocessed TLE sequences.

    Args:
        data_dir  : Directory containing sequence_*.npy files (aiml/data/)
        save_path : Where to store the trained model weights
        epochs    : Max training epochs
        batch_size: Mini-batch size
        lr        : Initial learning rate
        patience  : Early stopping patience (epochs without val improvement)

    Returns:
        Trained ResidualTrajectoryLSTM model
    """
    if data_dir is None:
        data_dir = os.path.join(AIML_ROOT, 'data')

    # ── Load splits ────────────────────────────────────────
    def _load(name):
        path = os.path.join(data_dir, f'sequence_{name}.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run tle_preprocessor.py first.")
        return torch.FloatTensor(np.load(path))

    X_train, y_train = _load('X_train'), _load('y_train')
    X_val,   y_val   = _load('X_val'),   _load('y_val')

    print("=" * 60)
    print("[LSTM] Training on real TLE sequences")
    print(f"  Train: {X_train.shape}  |  Val: {X_val.shape}")
    print("=" * 60)

    input_dim = X_train.shape[-1]  # 6 features
    model     = ResidualTrajectoryLSTM(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, verbose=True
    )

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    no_improve    = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)
        train_losses.append(epoch_loss)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs} | Train Loss: {epoch_loss:.5f} | Val Loss: {val_loss:.5f}")

        # ── Early stopping ────────────────────────────────────
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            no_improve    = 0
            # Save best checkpoint
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[LSTM] Early stopping at epoch {epoch+1} (no val improvement for {patience} epochs)")
                break

    # Save loss curves
    loss_path = os.path.join(os.path.dirname(save_path), 'lstm_loss.npy')
    np.save(loss_path, np.array(train_losses))

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    print(f"\n[OK] Best model saved → {save_path}  |  Best Val Loss: {best_val_loss:.5f}")
    return model


# ────────────────────────────────────────────────────────
def fine_tune(model:      'ResidualTrajectoryLSTM',
              data_dir:   str,
              save_path:  str,
              epochs:     int = 20,
              lr:         float = 5e-4) -> 'ResidualTrajectoryLSTM':
    """
    Fine-tune: freeze LSTM layers, retrain only the FC head.
    Use after training on synthetic data, then adapting to real data.
    """
    # Freeze LSTM
    for param in model.lstm.parameters():
        param.requires_grad = False

    def _load(name):
        return torch.FloatTensor(np.load(os.path.join(data_dir, f'sequence_{name}.npy')))

    X_train, y_train = _load('X_train'), _load('y_train')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.MSELoss()

    print(f"\n[LSTM] Fine-tuning FC head ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Fine-tune Epoch {epoch+1}/{epochs} | Loss: {loss.item():.5f}")

    # Unfreeze
    for param in model.lstm.parameters():
        param.requires_grad = True

    torch.save(model.state_dict(), save_path)
    print(f"[OK] Fine-tuned model saved → {save_path}")
    return model


# ────────────────────────────────────────────────────────
def evaluate(model: 'ResidualTrajectoryLSTM',
             data_dir: str = None) -> dict:
    """
    Evaluate model on test split.

    Returns:
        dict with 'rmse_km', 'mae_km', 'n_test'
    """
    if data_dir is None:
        data_dir = os.path.join(AIML_ROOT, 'data')

    X_test = torch.FloatTensor(np.load(os.path.join(data_dir, 'sequence_X_test.npy')))
    y_test = np.load(os.path.join(data_dir, 'sequence_y_test.npy'))

    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy()

    errors = preds - y_test
    rmse   = float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))
    mae    = float(np.mean(np.sqrt(np.sum(errors ** 2, axis=1))))

    print(f"\n[LSTM Evaluation]")
    print(f"  RMSE : {rmse:.4f} km")
    print(f"  MAE  : {mae:.4f} km")
    print(f"  N    : {len(y_test)}")
    return {'rmse_km': rmse, 'mae_km': mae, 'n_test': len(y_test)}


# ────────────────────────────────────────────────────────
def load_hybrid_model(path: str = 'models/hybrid_lstm.pth') -> 'ResidualTrajectoryLSTM':
    """Load the pre-trained Hybrid LSTM model."""
    model = ResidualTrajectoryLSTM()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        print(f"[LSTM] Loaded from {path}")
    else:
        print(f"[LSTM] Warning: model not found at {path} — using random weights")
    model.eval()
    return model


def predict_hybrid_correction(model: 'ResidualTrajectoryLSTM',
                               past_states: np.ndarray) -> np.ndarray:
    """
    Predict residual correction [dx, dy, dz] for the next timestep.

    Args:
        model       : Trained ResidualTrajectoryLSTM
        past_states : np.ndarray (seq_len, 6)  — [x,y,z,vx,vy,vz] history

    Returns:
        np.ndarray (3,)  — [dx, dy, dz] correction in km
    """
    model.eval()
    with torch.no_grad():
        x          = torch.FloatTensor(past_states).unsqueeze(0)
        correction = model(x)
    return correction.squeeze().numpy()


# ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train / evaluate LSTM model')
    parser.add_argument('--mode', choices=['synthetic', 'real', 'eval'], default='real',
                        help='Training mode')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path  = os.path.join(MODELS_DIR, 'hybrid_lstm.pth')
    data_dir   = os.path.join(AIML_ROOT, 'data')

    if args.mode == 'synthetic':
        train_hybrid_model(save_path=save_path, epochs=args.epochs)
    elif args.mode == 'real':
        model = train_on_real_data(data_dir=data_dir, save_path=save_path, epochs=args.epochs)
        evaluate(model, data_dir=data_dir)
    elif args.mode == 'eval':
        model = load_hybrid_model(save_path)
        evaluate(model, data_dir=data_dir)
