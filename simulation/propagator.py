import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sgp4.api import SatrecArray

class OrbitalPropagator:
    """
    Vectorized environment executing physics-based interactions 
    using SGP4 over large catalogs of bodies.
    """
    
    def __init__(self, tles: List[Dict]):
        self.tles = tles
        self.sate_array = SatrecArray([d['satrec'] for d in tles])
        self.sat_ids = np.array([d['id'] for d in tles])
        self.sat_names = np.array([d['name'] for d in tles])
        
    def _datetime_to_jd(self, dt: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """Convert standard python datetime to JD epoch required by SGP4."""
        JD_START = 2400000.5
        diff = dt - datetime(1858, 11, 17)
        jd_whole = np.array([diff.days + JD_START])
        jd_frac = np.array([diff.seconds / 86400.0 + diff.microseconds / 86400000000.0])
        return jd_whole, jd_frac

    def propagate(self, target_time: datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast vectorized physics calculation propagating the full catalog to the target epoch.
        Returns: 
           - valid_sat_ids (String Array)
           - valid_sat_names (String Array)
           - positions_km (X,Y,Z Array)
           - velocities_km_s (Vx,Vy,Vz Array)
        """
        jd, fr = self._datetime_to_jd(target_time)
        e, r, v = self.sate_array.sgp4(jd, fr)
        
        # Flatten and filter out objects that failed to propagate (like decaying orbits)
        e = e.flatten()
        r = r.reshape(-1, 3)
        v = v.reshape(-1, 3)
        
        valid_mask = (e == 0)
        return self.sat_ids[valid_mask], self.sat_names[valid_mask], r[valid_mask], v[valid_mask]

    def simulate_trajectory(self, start_time: datetime, duration_mins: int, step_size_mins: int = 1) -> Dict:
        """
        Generates continuous time-series trajectories for visualizing entire orbits.
        Creates a massive 3-dimensional trajectory map across time.
        """
        trajectories = {sat_id: {'positions': [], 'velocities': [], 'times': []} for sat_id in self.sat_ids}
        
        print(f"[Propagator] Simulating trajectory for {len(self.sat_ids)} bodies...")
        for minute_offset in range(0, duration_mins, step_size_mins):
            ts = start_time + timedelta(minutes=minute_offset)
            ids, _names, r, v = self.propagate(ts)
            
            for idx, sat_id in enumerate(ids):
                trajectories[sat_id]['positions'].append(r[idx])
                trajectories[sat_id]['velocities'].append(v[idx])
                trajectories[sat_id]['times'].append(ts)
                
        return trajectories

if __name__ == "__main__":
    from data_processor import DataProcessor
    import os
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processor = DataProcessor(
        tle_path=os.path.join(BASE_DIR, "data", "tle_data.txt"),
        conjunction_path=os.path.join(BASE_DIR, "data", "conjuction_and_constellation_data.csv")
    )
    sats = processor.load_tles()
    
    if sats:
        propagator = OrbitalPropagator(sats)
        now = datetime.utcnow()
        ids, names, pos, vel = propagator.propagate(now)
        print(f"Propagated {len(ids)} satellites successfully to Epoch {now}.")
