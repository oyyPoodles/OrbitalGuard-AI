"""
TLE Fetcher — Downloads and parses Two-Line Element sets from CelesTrak.
Falls back to local data/tle_data.txt if network unavailable.
"""
import os

TLE_URLS = [
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle",
]

LOCAL_TLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tle_data.txt')


def fetch_tle_online():
    """Attempt to download fresh TLE data from CelesTrak."""
    try:
        import urllib.request
        all_lines = []
        for url in TLE_URLS:
            response = urllib.request.urlopen(url, timeout=10)
            text = response.read().decode('utf-8')
            all_lines.extend(text.strip().splitlines())
        return all_lines
    except Exception as e:
        print(f"[Warning] Online TLE fetch failed: {e}")
        return None


def load_tle_lines(path=None):
    """
    Load TLE lines from local file.
    
    Args:
        path: Path to TLE file. Defaults to data/tle_data.txt.
        
    Returns:
        list of stripped lines.
    """
    path = path or LOCAL_TLE_PATH
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def parse_tle_entries(lines, max_objects=1500):
    """
    Parse raw TLE lines into structured entries.
    
    Args:
        lines: list of strings (name, line1, line2 repeating)
        max_objects: Maximum number of objects to parse.
        
    Returns:
        list of dicts: {name, line1, line2}
    """
    entries = []
    i = 0
    while i + 2 < len(lines) and len(entries) < max_objects:
        name = lines[i].strip()
        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()

        if line1.startswith('1') and line2.startswith('2'):
            entries.append({'name': name, 'line1': line1, 'line2': line2})
            i += 3
        else:
            i += 1  # skip malformed

    return entries


def get_tle_data(path=None, max_objects=1500):
    """
    Main entry point: load TLE data (online first, then local fallback).
    
    Returns:
        list of {name, line1, line2}
    """
    online = fetch_tle_online()
    if online and len(online) >= 3:
        print(f"[OK] Fetched {len(online)} TLE lines online")
        return parse_tle_entries(online, max_objects)

    print("[Info] Using local TLE data")
    lines = load_tle_lines(path)
    return parse_tle_entries(lines, max_objects)
