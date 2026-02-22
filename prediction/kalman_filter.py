import numpy as np

class OrbitalKalmanFilter:
    """
    Standard Linear Kalman Filter to smooth out noisy 3D Position tracking data
    from optical/radar sensors before feeding into the LSTM predictor.
    """
    
    def __init__(self, dt=1.0):
        # State vector: [x, y, z, vx, vy, vz]
        self.dt = dt
        self.state = np.zeros(6)
        
        # State Transition Matrix F (Kinematic mapping of position & velocity)
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt
        
        # Observation Matrix H (We only measure position [x, y, z])
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        
        # Initial Covariance Matrix P
        self.P = np.eye(6) * 1000
        
        # Process Noise Q (Model uncertainty)
        self.Q = np.eye(6) * 0.1
        
        # Measurement Noise R (Sensor uncertainty)
        self.R = np.eye(3) * 5.0
        
    def predict(self) -> np.ndarray:
        """Projects the state ahead in time."""
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.state[:3]
        
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Updates the state based on noisy measurement [x, y, z]."""
        # Innovation/Measurement residual
        y = measurement - np.dot(self.H, self.state)
        
        # Innovation covariance
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # Kalman Gain
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        
        # Map back to state
        self.state = self.state + np.dot(K, y)
        
        # Update covariance
        I = np.eye(6)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        return self.state[:3]

if __name__ == "__main__":
    # Test Kalman Filter
    kf = OrbitalKalmanFilter(dt=1.0)
    true_pos = np.array([100.0, 200.0, 300.0])
    noisy_pos = true_pos + np.random.normal(0, 5.0, 3)
    
    kf.predict()
    smoothed = kf.update(noisy_pos)
    print(f"True:   {true_pos}")
    print(f"Noisy:  {np.round(noisy_pos, 2)}")
    print(f"Smooth: {np.round(smoothed, 2)}")
