import numpy as np
from sgp4.api import SatrecArray

class Propagator:
    """Uses sgp4 SatrecArray to efficiently propagate hundreds of objects at once."""
    def __init__(self, objects):
        """objects: List of dicts from TLEParser"""
        self.objects = objects
        self.satrecs = [obj['satrec'] for obj in objects]
        self.array = SatrecArray(self.satrecs)
        
    def propagate(self, jd, fr):
        """
        Propagate all objects to a given Julian Date (jd) and fractional offset (fr).
        Returns arrays of positions (N, 3) and velocities (N, 3) in km and km/s.
        """
        e, r, v = self.array.sgp4(jd, fr)
        
        # sgp4 outputs r and v as (N, 1, 3) when given multiple jd inputs for a single time step.
        # We need to reshape them to (N, 3) for the collision and visualizers.
        if r.ndim == 3 and r.shape[1] == 1:
            r = r[:, 0, :]
            v = v[:, 0, :]
            
        return r, v
