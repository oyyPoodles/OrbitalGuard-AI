import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Try-except block for environment portability (RL training is heavy)
try:
    from stable_baselines3 import PPO
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

class SpaceAvoidanceEnv(gym.Env):
    """
    Custom Environment modeling a satellite trying to avoid an incoming debris object
    by firing thrusters (Delta-V).
    
    Observation Space: [Sat X, Sat Y, Sat Z, Debris X, Debris Y, Debris Z, Sat Vx, Sat Vy, Sat Vz]
    Action Space: [Delta-Vx, Delta-Vy, Delta-Vz] (Continuous thrust commands)
    """
    def __init__(self):
        super(SpaceAvoidanceEnv, self).__init__()
        
        # Actions: Accelerate in X, Y, Z between -1.0 and 1.0 km/s^2
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observations: 3D Pos/Vel of Sat, 3D Pos of Debris (9 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.max_steps = 100
        self.current_step = 0
        self.collision_threshold_km = 5.0
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize randomly: Sat at origin, debris 100km away barreling towards it
        self.sat_pos = np.array([0.0, 0.0, 0.0])
        self.sat_vel = np.array([7.0, 0.0, 0.0]) # Orbiting at 7km/s
        
        self.debris_pos = np.array([100.0, 10.0, 0.0])
        self.debris_vel = np.array([-10.0, -1.0, 0.0]) # Incoming fast
        
        return self._get_obs(), {}
        
    def _get_obs(self):
        return np.concatenate([self.sat_pos, self.debris_pos, self.sat_vel]).astype(np.float32)
        
    def step(self, action):
        self.current_step += 1
        
        # Apply Delta-V (Action) to Satellite Velocity
        self.sat_vel += action
        
        # Kinematics Update (Position += Velocity)
        self.sat_pos += self.sat_vel
        self.debris_pos += self.debris_vel
        
        # Calculate Miss Distance
        distance = np.linalg.norm(self.sat_pos - self.debris_pos)
        
        # Reward Function
        reward = 0.0
        terminated = False
        
        if distance < self.collision_threshold_km:
            # Massive penalty for collision
            reward = -1000.0
            terminated = True
        else:
            # Small penalty for using fuel (Delta-V norm), encourage efficiency
            fuel_cost = np.linalg.norm(action)
            reward = -fuel_cost
            
        # Success if we survive max_steps
        if self.current_step >= self.max_steps:
            terminated = True
            reward += 100.0 # Survival bonus
            
        return self._get_obs(), float(reward), terminated, False, {"miss_distance": distance}

class AvoidanceAgent:
    """Wrapper for running the trained RL Agent."""
    def __init__(self, model_path="models/ppo_avoidance.zip"):
        self.model_path = model_path
        self.env = SpaceAvoidanceEnv()
        
        if not RL_AVAILABLE:
            raise ImportError("[Avoidance] Stable-Baselines3 is required but not installed.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Avoidance] Trained PPO agent missing at {model_path}. Run train_rl.py first.")
            
        print(f"[Avoidance] Loading trained PPO agent from {model_path}...")
        self.model = PPO.load(model_path, env=self.env)
            
    def suggest_maneuver(self, state_obs: np.ndarray) -> np.ndarray:
        """Given the 9D State Matrix, returns the 3D Delta-V to apply."""
        action, _states = self.model.predict(state_obs, deterministic=True)
        return action

if __name__ == "__main__":
    agent = AvoidanceAgent()
    agent.train_or_mock(timesteps=2000) # Fast train for test
    
    obs, _ = agent.env.reset()
    action = agent.suggest_maneuver(obs)
    print(f"Observation State:\n{obs}")
    print(f"Suggested Delta-V Maneuver:\n{action}")
