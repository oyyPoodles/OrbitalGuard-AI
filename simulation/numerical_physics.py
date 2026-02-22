import numpy as np
from typing import Tuple

class OrbitalPhysicsDynamics:
    """
    Advanced Numerical Integrator for Spacecraft Maneuvers.
    SGP4 is analytical and excellent for TLE propagation, but cannot easily accept Delta-V inputs.
    This module uses Runge-Kutta 4 (RK4) integration with J2 and drag perturbations
    for accurate modeling of collision avoidance trajectories.
    """
    
    MU_EARTH = 398600.4418 # km^3 / s^2
    R_EARTH = 6378.137 # km
    J2 = 1.08262668e-3
    
    def __init__(self, dt_seconds: float = 1.0, cd: float = 2.2, area_m2: float = 1.0, mass_kg: float = 100.0):
        self.dt = dt_seconds
        self.cd = cd
        self.area_m2 = area_m2
        self.mass_kg = mass_kg
        
    def atmospheric_density(self, alt_km: float) -> float:
        """Simplified Exponential Atmospheric Density Model."""
        if alt_km > 1000.0:
            return 0.0
        # Base density (rho0) and scale height (H) approximation
        rho0 = 1.225e9 # kg/km^3 at sea level (1.225 kg/m^3)
        H = 8.5 # km
        return rho0 * np.exp(-alt_km / H)
        
    def gravity_and_perturbations(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Calculates acceleration vectors combining:
        1. Two-body point mass gravity
        2. J2 Earth Oblateness Perturbation
        3. Atmospheric Drag
        """
        r_norm = np.linalg.norm(r)
        
        # 1. Point Mass Gravity
        a_grav = (-self.MU_EARTH / (r_norm**3)) * r
        
        # 2. J2 Perturbation
        z2 = (r[2] / r_norm)**2
        j2_factor = (1.5 * self.J2 * self.MU_EARTH * (self.R_EARTH**2)) / (r_norm**5)
        
        a_j2 = np.zeros(3)
        a_j2[0] = j2_factor * r[0] * (5 * z2 - 1)
        a_j2[1] = j2_factor * r[1] * (5 * z2 - 1)
        a_j2[2] = j2_factor * r[2] * (5 * z2 - 3)
        
        # 3. Atmospheric Drag
        alt_km = r_norm - self.R_EARTH
        rho = self.atmospheric_density(alt_km)
        v_norm = np.linalg.norm(v)
        
        # F_drag = -0.5 * rho * v^2 * Cd * A * (v_dir)
        # a_drag = F_drag / m
        area_km2 = self.area_m2 / 1e6
        a_drag = -0.5 * rho * (v_norm) * self.cd * (area_km2 / self.mass_kg) * v
        
        return a_grav + a_j2 + a_drag

    def rk4_step(self, r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Runge-Kutta 4th Order continuous integration step."""
        
        k1_v = self.gravity_and_perturbations(r, v)
        k1_r = v
        
        k2_v = self.gravity_and_perturbations(r + 0.5 * self.dt * k1_r, v + 0.5 * self.dt * k1_v)
        k2_r = v + 0.5 * self.dt * k1_v
        
        k3_v = self.gravity_and_perturbations(r + 0.5 * self.dt * k2_r, v + 0.5 * self.dt * k2_v)
        k3_r = v + 0.5 * self.dt * k2_v
        
        k4_v = self.gravity_and_perturbations(r + self.dt * k3_r, v + self.dt * k3_v)
        k4_r = v + self.dt * k3_v
        
        r_next = r + (self.dt / 6.0) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        v_next = v + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return r_next, v_next

if __name__ == "__main__":
    physics = OrbitalPhysicsDynamics(dt_seconds=10.0)
    # LEO roughly 400km altitude
    r0 = np.array([physics.R_EARTH + 400.0, 0, 0])
    v0 = np.array([0, 7.67, 0]) # rough orbital velocity in km/s
    
    r_new, v_new = physics.rk4_step(r0, v0)
    print("Initial [R]:", r0)
    print("Initial [V]:", v0)
    print("Next [R]:   ", r_new)
    print("Next [V]:   ", v_new)
