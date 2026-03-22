"""
Environment — Assembles the full orbital simulation environment.
Combines TLE fetching + SGP4 propagation into a single interface.
"""
from datetime import datetime, timedelta
from simulation.tle_fetcher import get_tle_data
from simulation.sgp4_propagator import SGP4Propagator


class OrbitalEnvironment:
    def __init__(self, tle_path=None, max_objects=1500):
        """
        Initialize environment by loading TLE data and creating propagator.
        
        Args:
            tle_path: Optional path to local TLE file.
            max_objects: Maximum number of objects to track.
        """
        entries = get_tle_data(path=tle_path, max_objects=max_objects)
        self.propagator = SGP4Propagator(entries)
        self.sim_time = datetime.utcnow()
        self.object_count = len(self.propagator.satellites)
        print(f"🌍 Environment initialized with {self.object_count} objects")

    def step(self, dt_seconds=1.0):
        """
        Advance simulation by dt_seconds and return updated objects.
        
        Args:
            dt_seconds: Time step in seconds.
            
        Returns:
            list of object dicts with current position/velocity.
        """
        self.sim_time += timedelta(seconds=dt_seconds)
        return self.propagator.propagate(self.sim_time)

    def get_objects(self):
        """Return all tracked space objects with their current state."""
        return [
            {
                'id': s['id'],
                'name': s['name'],
                'position': s['position'].copy(),
                'velocity': s['velocity'].copy(),
                'type': s['type'],
            }
            for s in self.propagator.satellites
        ]

    def get_time(self):
        """Return current simulation timestamp."""
        return self.sim_time
