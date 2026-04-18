"""
SGP4 Propagator — Converts TLE entries into ECI position/velocity vectors.
"""
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime


def jday_from_datetime(dt):
    """Convert a datetime object to Julian Date (jd, fr)."""
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


class SGP4Propagator:
    def __init__(self, tle_entries):
        """
        Args:
            tle_entries: list of {name, line1, line2}
        """
        self.satellites = []
        for entry in tle_entries:
            try:
                sat = Satrec.twoline2rv(entry['line1'], entry['line2'])
                obj_id = entry['line1'][2:7].strip()

                # Comprehensive type classification from real TLE catalog patterns
                name_upper = entry['name'].upper().strip()
                if any(k in name_upper for k in (
                    'DEB', 'FRAG', 'FRAGMENT', 'DEBRIS',
                    'ARIANE DEB', 'SL-', 'FENGYUN', 'IRIDIUM 33 DEB',
                    'COSMOS 1408', 'COSMOS 2251 DEB', 'STARLINK DEB',
                )):
                    obj_type = 'debris'
                elif any(k in name_upper for k in (
                    'R/B', 'ROCKET', 'RKT', 'BODY', 'STAGE',
                    'BOOSTER', 'CENTAUR', 'H-2', 'ARIANE R/B',
                    'DELTA', 'ATLAS', 'PROTON', 'SOYUZ R/B', 'CZ-',
                )):
                    obj_type = 'rocket'
                else:
                    obj_type = 'payload'

                self.satellites.append({
                    'id': obj_id,
                    'name': entry['name'],
                    'satrec': sat,
                    'type': obj_type,
                    'position': np.zeros(3),
                    'velocity': np.zeros(3),
                })
            except Exception:
                pass

    def propagate(self, timestamp=None):
        """
        Propagate all satellites to a given timestamp.
        
        Args:
            timestamp: datetime object (defaults to now UTC)
            
        Returns:
            list of dicts with updated position/velocity
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        jd, fr = jday_from_datetime(timestamp)
        results = []

        for sat_data in self.satellites:
            sat = sat_data['satrec']
            e, r, v = sat.sgp4(jd, fr)

            if e == 0:
                sat_data['position'] = np.array(r)
                sat_data['velocity'] = np.array(v)
                results.append(sat_data)
            else:
                sat_data['position'] = np.array([np.nan, np.nan, np.nan])
                sat_data['velocity'] = np.array([np.nan, np.nan, np.nan])

        return results

    def propagate_single(self, sat_data, timestamp):
        """Propagate a single satellite object."""
        jd, fr = jday_from_datetime(timestamp)
        e, r, v = sat_data['satrec'].sgp4(jd, fr)
        if e == 0:
            return np.array(r), np.array(v)
        return None, None
