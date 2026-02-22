from removal_logic.simulation.tle_parser import TLEParser
from removal_logic.simulation.propagator import Propagator
from datetime import datetime
import numpy as np

class SpaceEnvironment:
    def __init__(self, tle_path):
        parser = TLEParser(tle_path)
        all_tles = parser.parse()
        
        self.active_tles, self.debris_tles = [], []
        for tle in all_tles:
            name = tle["name"].upper()
            if "DEB" in name or "R/B" in name or "OBJECT" in name:
                self.debris_tles.append(tle)
            else:
                self.active_tles.append(tle)
                
        print(f"[DEBUG] Filtered from Catalog: {len(self.active_tles)} Active Satellites | {len(self.debris_tles)} Debris Fields.")
        
        # Keep manageable sizes for dashboard rendering speed
        self.active_tles = self.active_tles[:500]
        self.debris_tles = self.debris_tles[:500]
        
        if len(self.debris_tles) == 0:
            print("[DEBUG] WARNING: 0 Debris found. Generating synthetic debris for simulation.")
            synth_deb = {"name": "SYNTHETIC DEBRIS", "line1": "1 99999U 00000A   26053.00000000  .00000000  00000-0  00000-0 0  9999", "line2": "2 99999  90.0000   0.0000 0000000   0.0000   0.0000 14.00000000    05"}
            self.debris_tles = [synth_deb]
            
        if len(self.active_tles) == 0:
            print("[DEBUG] WARNING: 0 Active found. Generating synthetic satellite for simulation.")
            synth_act = {"name": "SYNTHETIC SAT", "line1": "1 99998U 00000B   26053.00000000  .00000000  00000-0  00000-0 0  9998", "line2": "2 99998  45.0000  90.0000 0000000  90.0000 180.0000 13.00000000    04"}
            self.active_tles = [synth_act]
            
        print(f"[DEBUG] Propagating {len(self.active_tles)} Active and {len(self.debris_tles)} Debris via SGP4.")
        self.active_propagator = Propagator(self.active_tles)
        self.debris_propagator = Propagator(self.debris_tles)
        
        self.active_positions = np.array([])
        self.debris_positions = np.array([])
        self.active_velocities = np.array([])
        self.debris_velocities = np.array([])
        
        self.rocket_active = False
        self.rocket_pos = np.array([0.0, 0.0, 0.0])
        self.rocket_target_idx = None
        
        # Populate arrays to avoid initial empty render
        self.step_simulation(datetime.utcnow())
        
    def _datetime_to_jd(self, dt):
        year, month, day = dt.year, dt.month, dt.day
        hour, min, sec = dt.hour, dt.minute, dt.second + dt.microsecond / 1e6
        if month <= 2:
            year -= 1; month += 12
        A = int(year / 100)
        B = 2 - A + int(A / 4)
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
        fr = (hour + min / 60.0 + sec / 3600.0) / 24.0
        return jd, fr
        
    def step_simulation(self, current_time):
        jd, fr = self._datetime_to_jd(current_time)
        a_pos, a_vel, _ = self.active_propagator.get_positions(jd, fr)
        d_pos, d_vel, _ = self.debris_propagator.get_positions(jd, fr)
        
        # Fallback if propagation completely fails resulting in empty valid arrays
        if len(a_pos) == 0:
            print("[DEBUG] WARNING: SGP4 returned 0 valid Active positions. Injecting fallback.")
            a_pos = np.array([[7000.0, 0.0, 0.0]])
        if len(d_pos) == 0:
            print("[DEBUG] WARNING: SGP4 returned 0 valid Debris positions. Injecting fallback.")
            d_pos = np.array([[0.0, 7000.0, 0.0]])
            
        self.active_positions, self.debris_positions = a_pos, d_pos
        self.active_velocities, self.debris_velocities = a_vel, d_vel
        
    def get_state(self):
        return {
            "active_satellites": self.active_positions,
            "debris": self.debris_positions,
            "rocket_active": self.rocket_active,
            "rocket_pos": self.rocket_pos
        }
        
    def remove_debris(self, index):
        if 0 <= index < len(self.debris_propagator.satellites):
            del self.debris_propagator.satellites[index]
