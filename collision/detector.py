import numpy as np

class CollisionDetector:
    """Vectorized Euclidean distance calculations with altitude filtering."""
    def __init__(self, state_manager):
        self.state = state_manager
        
    def detect_risks(self):
        """
        Calculates all pairwise distances. Uses altitude banding to ignore faraway objects.
        Returns a list of dicts describing active risks.
        """
        risks = []
        pos = self.state.positions
        vel = self.state.velocities
        objs = self.state.objects
        
        if pos is None or len(pos) < 2:
            return risks
            
        N = len(pos)
        
        # Calculate altitudes from earth center (magnitude of position vector)
        altitudes = np.linalg.norm(pos, axis=1)
        
        for i in range(N):
            # Skip invalid positions
            if np.isnan(altitudes[i]):
                continue
                
            for j in range(i + 1, N): # Check unique pairs
                if np.isnan(altitudes[j]):
                    continue
                    
                # 1. Altitude Band Filtering
                # If they are more than 50km apart in raw altitude, skip exact distance calculation
                if abs(altitudes[i] - altitudes[j]) > 50.0:
                    continue
                    
                # 2. Euclidean Distance Calculation
                dist = np.linalg.norm(pos[i] - pos[j])
                
                # 3. Relative Velocity
                rel_vel = np.linalg.norm(vel[i] - vel[j])
                
                # 4. Risk Classification
                risk_level = None
                if dist < 1.0:
                    risk_level = "HIGH"
                elif dist < 5.0:
                    risk_level = "MEDIUM"
                
                if risk_level:
                    risks.append({
                        'obj1_id': objs[i]['id'],
                        'obj2_id': objs[j]['id'],
                        'obj1_name': objs[i]['name'],
                        'obj2_name': objs[j]['name'],
                        'distance_km': dist,
                        'relative_velocity_km_s': rel_vel,
                        'risk_level': risk_level
                    })
                    
        # Sort by most critical distance
        risks.sort(key=lambda x: x['distance_km'])
        return risks
