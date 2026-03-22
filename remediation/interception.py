"""
STEP 7: Autonomous Debris Interception Simulation
Selects nearest debris and computes a dynamic interception trajectory.
"""
import numpy as np


class InterceptionSimulator:
    def __init__(self, intercept_speed=2.0):
        """
        Args:
            intercept_speed: Speed of the interceptor (km/s).
        """
        self.intercept_speed = intercept_speed

    def select_target(self, interceptor_pos, debris_objects):
        """
        Select the nearest debris object for interception.
        
        Args:
            interceptor_pos: np.array (3,) — current interceptor position
            debris_objects: list of dicts with 'name', 'position'
            
        Returns:
            Target dict or None.
        """
        if not debris_objects:
            return None

        min_dist = float('inf')
        target = None

        for obj in debris_objects:
            pos = np.array(obj['position'])
            if np.any(np.isnan(pos)):
                continue
            d = np.linalg.norm(pos - interceptor_pos)
            if d < min_dist:
                min_dist = d
                target = obj

        return target

    def compute_intercept_trajectory(self, start_pos, target_pos, steps=50):
        """
        Compute a dynamic interception path from start to target.
        
        Args:
            start_pos: np.array (3,) — launch position
            target_pos: np.array (3,) — target debris position
            steps: Number of trajectory waypoints
            
        Returns:
            list of np.array (3,) waypoints
        """
        start = np.array(start_pos, dtype=float)
        target = np.array(target_pos, dtype=float)
        
        trajectory = []
        for i in range(steps + 1):
            t = i / steps
            # Linear interpolation with slight curve
            point = start + t * (target - start)
            # Add sinusoidal perturbation for realistic curvature
            point += np.sin(t * np.pi) * np.array([50, 30, 20]) * (1 - t)
            trajectory.append(point)
            
        return trajectory

    def simulate_mission(self, interceptor_pos, debris_objects):
        """
        Full interception mission: select target → compute path → return result.
        
        Returns:
            dict with 'target', 'trajectory', 'distance', 'eta_seconds'
        """
        target = self.select_target(interceptor_pos, debris_objects)
        if target is None:
            return {'status': 'NO_TARGET'}

        target_pos = np.array(target['position'])
        trajectory = self.compute_intercept_trajectory(interceptor_pos, target_pos)
        distance = np.linalg.norm(target_pos - interceptor_pos)
        eta = distance / self.intercept_speed if self.intercept_speed > 0 else float('inf')

        return {
            'status': 'INTERCEPT_PLANNED',
            'target': target.get('name', 'UNKNOWN'),
            'distance_km': round(float(distance), 2),
            'eta_seconds': round(float(eta), 2),
            'trajectory': trajectory,
            'waypoints': len(trajectory)
        }
