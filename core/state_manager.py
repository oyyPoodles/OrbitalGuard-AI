import datetime
import numpy as np
from collections import deque
from sgp4.api import jday

class StateManager:
    """Manages global simulation time, current states, and rendering history."""
    def __init__(self, propagator):
        self.propagator = propagator
        self.objects = propagator.objects
        self.current_time = datetime.datetime.utcnow()
        self.positions = None
        self.velocities = None
        
        # Store trail history: dict of deque arrays
        self.history_length = 15 # Store past 15 positions to draw a trail line
        self.history = {obj['id']: deque(maxlen=self.history_length) for obj in self.objects}
        
    def step(self, dt_seconds):
        """Advances time by dt_seconds and updates all positions and history."""
        self.current_time += datetime.timedelta(seconds=dt_seconds)
        jd, fr = jday(
            self.current_time.year, self.current_time.month, self.current_time.day,
            self.current_time.hour, self.current_time.minute, self.current_time.second
        )
        # SGP4 Array Propagation requires numpy arrays
        # Pass a 1-element array so all satellites calculate against this single time frame
        jd_arr = np.array([jd])
        fr_arr = np.array([fr])
        r, v = self.propagator.propagate(jd_arr, fr_arr)
        self.positions = r
        self.velocities = v
        
        # Update Trail History
        for i, obj in enumerate(self.objects):
            if not np.isnan(r[i][0]): # Ensure valid coordinate
                self.history[obj['id']].append(r[i])
