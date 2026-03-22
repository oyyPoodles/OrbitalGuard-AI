"""
STEP 2: 6D Kalman Filter for Orbital State Estimation
State vector: [x, y, z, vx, vy, vz]
Reduces measurement noise from the detection layer.
"""
import numpy as np

class KalmanFilter6D:
    def __init__(self, dt=1.0, process_noise=0.01, measurement_noise=0.5):
        """
        Args:
            dt: Time step (seconds).
            process_noise: Process noise covariance scalar.
            measurement_noise: Measurement noise covariance scalar.
        """
        self.dt = dt
        self.n = 6  # state dimension

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.n)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix (observe all 6 states)
        self.H = np.eye(self.n)

        # Process noise covariance
        self.Q = np.eye(self.n) * process_noise

        # Measurement noise covariance
        self.R = np.eye(self.n) * measurement_noise

        # State estimate and covariance
        self.x = np.zeros(self.n)
        self.P = np.eye(self.n) * 1.0

        self.initialized = False

    def initialize(self, state):
        """Set initial state vector [x, y, z, vx, vy, vz]."""
        self.x = np.array(state, dtype=float)
        self.initialized = True

    def predict(self):
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, measurement):
        """
        Update state with new measurement.
        
        Args:
            measurement: np.array of shape (6,) — [x, y, z, vx, vy, vz]
            
        Returns:
            Refined state vector.
        """
        if not self.initialized:
            self.initialize(measurement)
            return self.x.copy()

        z = np.array(measurement, dtype=float)

        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y
        self.P = (np.eye(self.n) - K @ self.H) @ self.P

        return self.x.copy()

    def filter_sequence(self, measurements):
        """
        Process a full sequence of measurements.
        
        Args:
            measurements: np.array of shape (N, 6)
            
        Returns:
            np.array of filtered states (N, 6)
        """
        filtered = []
        for z in measurements:
            self.predict()
            state = self.update(z)
            filtered.append(state)
        return np.array(filtered)
