import os
import torch
import torch.nn as nn
import numpy as np

class TrajectoryLSTM(nn.Module):
    """
    LSTM sequence-to-sequence model to predict future orbital states
    based on historical filtered sensor observations.
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, output_dim=3):
        super(TrajectoryLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # We only care about the prediction at the last time step
        out = self.fc(out[:, -1, :])
        return out

class TrajectoryPredictor:
    """Wrapper that handles data scaling and model inference."""
    def __init__(self, model_path="models/lstm_trajectory.pth"):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TrajectoryLSTM().to(self.device)
        
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"[LSTM] Trained model weights missing at {model_path}. Run train_lstm.py first.")
             
        print(f"[LSTM] Loading real trained sequences from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
            
    def predict_next_position(self, history_sequence_km: np.ndarray) -> np.ndarray:
        """
        Takes shape (Seq_Len, 3) where columns are X, Y, Z.
        Outputs predicted next (X, Y, Z) via LSTM.
        """
        # Shape prep: (Batch, Seq, Features)
        seq_tensor = torch.tensor(history_sequence_km, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(seq_tensor)
            
        return prediction.cpu().numpy().squeeze()

if __name__ == "__main__":
    predictor = TrajectoryPredictor()
    
    # Simulate a history of 10 coordinates moving in a straight line
    mock_history = np.array([
        [10.0, 20.0, 30.0],
        [11.0, 21.0, 31.0],
        [12.0, 22.0, 32.0],
        [13.0, 23.0, 33.0],
        [14.0, 24.0, 34.0],
    ])
    
    next_pos = predictor.predict_next_position(mock_history)
    print("Historical sequence head:\n", mock_history[:2])
    print("Predicted future offset:\n", next_pos)
