import os

class TLEParser:
    def __init__(self, file_path):
        self.file_path = file_path
        
    def parse(self):
        satellites = []
        if not os.path.exists(self.file_path):
            print(f"[DEBUG] TLE data file not found at {self.file_path}")
            return satellites
            
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        print(f"[DEBUG] Read {len(lines)} non-empty lines from TLE file {self.file_path}")
        
        name = "UNKNOWN"
        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("1 ") and i + 1 < len(lines) and lines[i+1].startswith("2 "):
                l1 = line
                l2 = lines[i+1]
                satellites.append({"name": name, "line1": l1, "line2": l2})
            elif not line.startswith("1 ") and not line.startswith("2 "):
                name = line
                
        print(f"[DEBUG] Successfully parsed {len(satellites)} satellites from TLE data.")
        return satellites
