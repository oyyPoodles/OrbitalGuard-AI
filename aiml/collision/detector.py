"""
STEP 4: KDTree-based Collision Detection
Efficiently identifies close-approach object pairs using spatial indexing.
"""
import numpy as np
from scipy.spatial import KDTree

def compute_distance(p1, p2):
    """Euclidean distance between two position vectors."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def relative_velocity(v1, v2):
    """Relative velocity magnitude between two velocity vectors."""
    return np.linalg.norm(np.array(v1) - np.array(v2))


class CollisionDetector:
    def __init__(self, threshold_km=5.0):
        """
        Args:
            threshold_km: Maximum distance (km) to flag as a close approach.
        """
        self.threshold_km = threshold_km

    def detect(self, objects):
        """
        Detect close-approach pairs using KDTree.
        
        Args:
            objects: list of dicts with 'id', 'name', 'position', 'velocity', 'type'
            
        Returns:
            list of conjunction event dicts.
        """
        valid = [o for o in objects if not np.any(np.isnan(o['position']))]
        if len(valid) < 2:
            return []

        positions = np.array([o['position'] for o in valid])

        # O(N log N) spatial query
        tree = KDTree(positions)
        pairs = tree.query_pairs(r=self.threshold_km)

        conjunctions = []
        for (i, j) in pairs:
            o1, o2 = valid[i], valid[j]
            d = compute_distance(o1['position'], o2['position'])
            v = relative_velocity(o1['velocity'], o2['velocity'])

            conjunctions.append({
                'obj1_id': o1.get('id', o1.get('name', str(i))),
                'obj2_id': o2.get('id', o2.get('name', str(j))),
                'distance_km': round(float(d), 4),
                'relative_velocity_kms': round(float(v), 4),
                'obj1_type': o1.get('type', 'unknown'),
                'obj2_type': o2.get('type', 'unknown'),
            })

        return conjunctions
