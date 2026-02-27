import os
from sgp4.api import Satrec, WGS72

class TLEParser:
    """Parses raw TLE data and initializes SGP4 Satrec objects."""
    def __init__(self, tle_path):
        self.tle_path = tle_path
        
    def load_objects(self, max_objects=300):
        """
        Reads TLE file, creates Satrec objects, and distinguishes debris from active satellites.
        Returns a list of dicts: [{'id': str, 'name': str, 'type': 'SAT'|'DEB', 'satrec': Satrec}, ...]
        """
        if not os.path.exists(self.tle_path):
            raise FileNotFoundError(f"TLE data file not found at {self.tle_path}")
            
        objects = []
        with open(self.tle_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Parse groups of 3 (Name, Line 1, Line 2)
        count = 0
        for i in range(0, len(lines), 3):
            if count >= max_objects:
                break
                
            if i + 2 < len(lines):
                name = lines[i]
                line1 = lines[i+1]
                line2 = lines[i+2]
                
                # Basic name heuristic for classification
                obj_type = "DEB" if "DEB" in name.upper() or "R/B" in name.upper() else "SAT"
                
                try:
                    satrec = Satrec.twoline2rv(line1, line2)
                    # Extract catalog ID from line 1
                    cat_id = line1[2:7].strip()
                    
                    objects.append({
                        'id': cat_id,
                        'name': name,
                        'type': obj_type,
                        'satrec': satrec
                    })
                    count += 1
                except Exception as e:
                    print(f"Error parsing {name}: {e}")
                    
        return objects
