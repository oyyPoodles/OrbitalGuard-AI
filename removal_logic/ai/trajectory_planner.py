import numpy as np

class TrajectoryPlanner:
    def __init__(self, rocket_pos, target_pos, speed=50.0):
        self.rocket_pos = rocket_pos
        self.target_pos = target_pos
        self.speed = speed
        
    def step(self, dt_seconds):
        direction = self.target_pos - self.rocket_pos
        distance = np.linalg.norm(direction)
        
        if distance < 10.0:
            return self.target_pos, True
            
        direction_norm = direction / distance
        displacement = direction_norm * self.speed * dt_seconds
        
        if np.linalg.norm(displacement) > distance:
            self.rocket_pos = self.target_pos
            return self.rocket_pos, True
            
        self.rocket_pos = self.rocket_pos + displacement
        return self.rocket_pos, False
