"""
STEP 6: PPO Reinforcement Learning Agent for Collision Avoidance
Action: ΔV thrust vector (3D)
Reward: avoid collision + minimize fuel expenditure
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️ stable-baselines3 not installed. PPO agent will use rule-based fallback.")


class CollisionAvoidanceEnv(gym.Env):
    """Custom Gym environment for satellite collision avoidance."""
    
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # Action: ΔV vector (3D thrust), normalized [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # Observation: [relative_pos(3), relative_vel(3), own_pos(3), own_vel(3)] = 12
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.max_steps = 50
        self.step_count = 0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Random initial state
        self.sat_pos = np.random.uniform(-100, 100, 3).astype(np.float32)
        self.sat_vel = np.random.uniform(-1, 1, 3).astype(np.float32)
        self.debris_pos = self.sat_pos + np.random.uniform(-10, 10, 3).astype(np.float32)
        self.debris_vel = np.random.uniform(-1, 1, 3).astype(np.float32)
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        rel_pos = self.debris_pos - self.sat_pos
        rel_vel = self.debris_vel - self.sat_vel
        return np.concatenate([rel_pos, rel_vel, self.sat_pos, self.sat_vel]).astype(np.float32)

    def step(self, action):
        # Apply ΔV
        dv = action * 0.1  # scale thrust
        self.sat_vel += dv

        # Advance positions
        self.sat_pos += self.sat_vel
        self.debris_pos += self.debris_vel

        self.step_count += 1

        # Compute reward
        dist = np.linalg.norm(self.sat_pos - self.debris_pos)
        fuel_cost = np.linalg.norm(dv)

        reward = 0.0
        terminated = False
        truncated = False

        if dist < 1.0:
            reward = -100.0  # collision penalty
            terminated = True
        elif dist > 50.0:
            reward = 10.0  # safely avoided
            terminated = True
        else:
            reward = dist * 0.1 - fuel_cost * 5.0  # reward distance, penalize fuel

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}


class PPOAvoidanceAgent:
    def __init__(self, model_path='models/ppo_avoidance.zip'):
        self.model_path = model_path
        self.model = None
        self.env = CollisionAvoidanceEnv()

        if HAS_SB3:
            try:
                self.model = PPO.load(model_path, env=self.env)
            except:
                pass

    def train(self, total_timesteps=10000):
        """Train PPO agent."""
        if not HAS_SB3:
            print("Cannot train: stable-baselines3 not installed.")
            return
        
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=total_timesteps)
        self.model.save(self.model_path)
        print(f"[OK] PPO agent saved to {self.model_path}")

    def compute_avoidance(self, relative_position, relative_velocity, own_position, own_velocity):
        """
        Compute optimal ΔV maneuver.
        
        Returns:
            dict with 'delta_v' (np.array of shape (3,)) and 'fuel_cost' (float)
        """
        obs = np.concatenate([relative_position, relative_velocity, own_position, own_velocity]).astype(np.float32)

        if self.model and HAS_SB3:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            # Rule-based fallback: thrust away from debris
            direction = -relative_position / (np.linalg.norm(relative_position) + 1e-8)
            action = direction * 0.5

        dv = action * 0.1
        return {
            'delta_v': dv,
            'fuel_cost': float(np.linalg.norm(dv))
        }
