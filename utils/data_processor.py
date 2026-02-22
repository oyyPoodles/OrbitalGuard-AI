import os
import pandas as pd
from typing import List, Dict
from sgp4.api import Satrec, WGS72

class DataProcessor:
    """
    Handles data ingestion, TLE parsing, and structuring CSV data
    for the rest of the AI pipeline.
    """
    def __init__(self, tle_path: str, conjunction_path: str):
        self.tle_path = tle_path
        self.conjunction_path = conjunction_path
        
    def load_tles(self) -> List[Dict]:
        """Parses the raw TLE dataset into SGP4 Satrec objects."""
        parsed_sats = []
        if not os.path.exists(self.tle_path):
            print(f"Warning: TLE file {self.tle_path} not found.")
            return []
            
        with open(self.tle_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"[Data] Read {len(lines)} non-empty lines from TLE file {self.tle_path}")
        
        seen_ids = set()
        name = "UNKNOWN"
        
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("1 ") and i + 1 < len(lines) and lines[i+1].startswith("2 "):
                line1 = line
                line2 = lines[i+1]
                
                try:
                    sat_id = line1[2:7].strip()
                    if sat_id in seen_ids:
                        continue
                    seen_ids.add(sat_id)
                    
                    sat = Satrec.twoline2rv(line1, line2)
                    parsed_sats.append({
                        "id": sat_id,
                        "name": name,
                        "line1": line1,
                        "line2": line2,
                        "satrec": sat
                    })
                except Exception:
                    pass
            elif not line.startswith("1 ") and not line.startswith("2 "):
                name = line
                
        # Fallback to prevent hard UI crashes if the file data is completely invalid
        if len(parsed_sats) == 0:
            print("[Data] ERROR: 0 valid TLEs found. Injecting synthetic fallback satellites.")
            try:
                # Inject 3 static synthetic bodies to keep pipeline alive
                synth1 = Satrec.twoline2rv("1 99999U 00000A   26053.00000000  .00000000  00000-0  00000-0 0  9999", "2 99999  90.0000   0.0000 0000000   0.0000   0.0000 14.00000000    05")
                synth2 = Satrec.twoline2rv("1 99998U 00000B   26053.00000000  .00000000  00000-0  00000-0 0  9998", "2 99998  45.0000  90.0000 0000000  90.0000 180.0000 13.00000000    04")
                synth3 = Satrec.twoline2rv("1 99997U 00000C   26053.00000000  .00000000  00000-0  00000-0 0  9997", "2 99997   0.0000 180.0000 0000000 180.0000 270.0000 12.00000000    03")
                parsed_sats = [
                    {"id": "99999", "name": "SYNTH_1", "line1": "", "line2": "", "satrec": synth1},
                    {"id": "99998", "name": "SYNTH_2", "line1": "", "line2": "", "satrec": synth2},
                    {"id": "99997", "name": "SYNTH_3", "line1": "", "line2": "", "satrec": synth3}
                ]
            except Exception as e:
                print(f"Fallback generation also failed: {e}")
                
        print(f"[Data] Loaded {len(parsed_sats)} valid TLEs into global pipeline.")
        return parsed_sats

    def load_conjunctions(self) -> pd.DataFrame:
        """Loads and cleans the initial conjunction dataset layout."""
        if not os.path.exists(self.conjunction_path):
            print(f"Warning: Conjunction file {self.conjunction_path} not found.")
            return pd.DataFrame()
            
        df = pd.read_csv(self.conjunction_path)
        print(f"[Data] Loaded {len(df)} conjunction records.")
        return df
