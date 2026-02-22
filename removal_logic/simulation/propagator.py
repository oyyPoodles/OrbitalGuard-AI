import numpy as np
from sgp4.api import Satrec

class Propagator:
    def __init__(self, tles):
        self.satellites = []
        for tle in tles:
            try:
                sat = Satrec.twoline2rv(tle["line1"], tle["line2"])
                self.satellites.append({"name": tle["name"], "model": sat})
            except Exception as e:
                pass
        print(f"[DEBUG] Propagator initialized with {len(self.satellites)} valid SGP4 models out of {len(tles)} TLEs.")
                
    def get_positions(self, jd, fr):
        if len(self.satellites) == 0:
            return np.array([]), np.array([]), []
            
        positions, velocities, valid_indices = [], [], []
        for i, sat_data in enumerate(self.satellites):
            e, r, v = sat_data["model"].sgp4(jd, fr)
            if e == 0:
                positions.append(r)
                velocities.append(v)
                valid_indices.append(i)
        
        return np.array(positions), np.array(velocities), valid_indices
        
    def get_active_satellites(self):
        return self.satellites
