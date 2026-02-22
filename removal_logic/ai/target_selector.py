import numpy as np

class TargetSelector:
    def __init__(self, debris_positions, rocket_pos):
        self.debris_positions = debris_positions
        self.rocket_pos = rocket_pos
        
    def select_highest_risk(self):
        if len(self.debris_positions) == 0:
            return None, float('inf')
        distances = np.linalg.norm(self.debris_positions - self.rocket_pos, axis=1)
        best_idx = np.argmin(distances)
        return best_idx, distances[best_idx]
