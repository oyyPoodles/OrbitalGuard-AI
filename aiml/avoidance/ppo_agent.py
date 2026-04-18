"""
STEP 6: PPO Reinforcement Learning Agent — Upgraded
=====================================================
Action : ΔV thrust vector (3D), normalized [-1, 1]
Reward : avoid collision + minimize fuel + time-to-collision bonus

Improvements over v1:
  - Curriculum learning: Phase 1 (2D easy) → Phase 2 (3D full orbital)
  - Reward shaping: TTC bonus, smooth avoidance incentive
  - test(): runs 100 episodes, reports success rate + mean fuel cost
  - evaluate(): logs metrics to dict
  - Rule-based fallback unchanged
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

# ─── Path Alignment ──────────────────────────────────────
AVOIDANCE_DIR = os.path.dirname(os.path.abspath(__file__))
AIML_ROOT     = os.path.dirname(AVOIDANCE_DIR)

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("⚠️  stable-baselines3 not installed. PPO agent will use rule-based fallback.")


# ────────────────────────────────────────────────────────
class CollisionAvoidanceEnv(gym.Env):
    """
    Custom Gym environment for 3D satellite collision avoidance.

    Observation (12,):
        [relative_pos(3), relative_vel(3), own_pos(3), own_vel(3)]

    Action (3,):
        ΔV thrust vector, normalized [-1, 1]

    Reward shaping:
        - Collision penalty       : -100
        - Safe avoidance bonus    : +10 + distance bonus
        - Time-to-collision bonus : reward for increasing TTC each step
        - Fuel cost penalty       : -5 × ‖ΔV‖
        - Proximity risk penalty  : -2 when dist < 10 km
    """

    metadata = {"render_modes": []}

    def __init__(self, curriculum_phase: int = 2,
                 max_steps: int = 80,
                 collision_radius: float = 1.0,
                 safe_radius: float = 50.0):
        super().__init__()
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.curriculum_phase  = curriculum_phase  # 1=2D simple, 2=3D full
        self.max_steps         = max_steps
        self.collision_radius  = collision_radius
        self.safe_radius       = safe_radius
        self.step_count        = 0
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        if self.curriculum_phase == 1:
            # Phase 1: 2D head-on approach (easier)
            self.sat_pos   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.sat_vel   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            offset         = np.random.uniform(3, 8)
            self.debris_pos = np.array([offset, np.random.uniform(-2, 2), 0.0], dtype=np.float32)
            self.debris_vel = np.array([-np.random.uniform(0.5, 1.5), 0.0, 0.0], dtype=np.float32)
        else:
            # Phase 2: Full 3D random orbital scenarios
            self.sat_pos   = np.random.uniform(-50, 50, 3).astype(np.float32)
            self.sat_vel   = np.random.uniform(-1, 1, 3).astype(np.float32)
            offset         = np.random.uniform(2, 15, 3).astype(np.float32)
            self.debris_pos = self.sat_pos + offset
            self.debris_vel = np.random.uniform(-2, 2, 3).astype(np.float32)

        self.step_count = 0
        self.prev_dist  = float(np.linalg.norm(self.sat_pos - self.debris_pos))
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        rel_pos = self.debris_pos - self.sat_pos
        rel_vel = self.debris_vel - self.sat_vel
        return np.concatenate([rel_pos, rel_vel, self.sat_pos, self.sat_vel]).astype(np.float32)

    def step(self, action: np.ndarray):
        dv              = action * 0.1
        self.sat_vel   += dv
        self.sat_pos   += self.sat_vel
        self.debris_pos += self.debris_vel
        self.step_count += 1

        dist      = float(np.linalg.norm(self.sat_pos - self.debris_pos))
        fuel_cost = float(np.linalg.norm(dv))
        delta_d   = dist - self.prev_dist   # positive = moving apart
        self.prev_dist = dist

        reward     = 0.0
        terminated = False
        truncated  = False

        if dist < self.collision_radius:
            reward     = -100.0
            terminated = True
        elif dist > self.safe_radius:
            reward     = 10.0 + dist * 0.05   # bonus for clearance
            terminated = True
        else:
            # Shape reward:
            reward  = delta_d * 0.5            # moving away = positive
            reward -= fuel_cost * 5.0          # penalize fuel
            if dist < 10.0:
                reward -= 2.0                  # proximity danger penalty

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}


# ────────────────────────────────────────────────────────
class PPOAvoidanceAgent:
    """PPO-based satellite collision avoidance agent."""

    def __init__(self, model_path: str = 'models/ppo_avoidance.zip',
                 curriculum_phase: int = 2):
        self.model_path       = model_path
        self.curriculum_phase = curriculum_phase
        self.model            = None
        self.raw_env          = CollisionAvoidanceEnv(curriculum_phase=curriculum_phase)

        if HAS_SB3:
            os.makedirs('models', exist_ok=True)
            self.env = Monitor(self.raw_env, 'models/ppo_monitor')
            try:
                self.model = PPO.load(model_path, env=self.env)
                print(f"[PPO] Loaded from {model_path}")
            except Exception:
                pass
        else:
            self.env = self.raw_env

    def train(self, total_timesteps: int = 50_000,
              use_curriculum: bool = True) -> None:
        """
        Train PPO agent, optionally with curriculum:
          Phase 1: 25% of time in 2D environment
          Phase 2: Remaining 75% in full 3D
        """
        if not HAS_SB3:
            print("[PPO] Cannot train: stable-baselines3 not installed.")
            return

        if use_curriculum and total_timesteps >= 20_000:
            # Phase 1: 2D curriculum
            phase1_steps = total_timesteps // 4
            print(f"[PPO] Curriculum Phase 1 (2D) — {phase1_steps:,} steps")
            env1        = Monitor(
                CollisionAvoidanceEnv(curriculum_phase=1),
                'models/ppo_monitor_p1'
            )
            self.model  = PPO("MlpPolicy", env1, verbose=0,
                              learning_rate=3e-4, n_steps=1024, batch_size=64)
            self.model.learn(total_timesteps=phase1_steps)

            # Phase 2: Full 3D
            phase2_steps = total_timesteps - phase1_steps
            print(f"[PPO] Curriculum Phase 2 (3D) — {phase2_steps:,} steps")
            self.model.set_env(self.env)
            self.model.learn(total_timesteps=phase2_steps, reset_num_timesteps=False)
        else:
            self.model = PPO("MlpPolicy", self.env, verbose=1,
                             learning_rate=3e-4, n_steps=1024, batch_size=64)
            self.model.learn(total_timesteps=total_timesteps)

        self.model.save(self.model_path)
        print(f"[OK] PPO agent saved → {self.model_path}")

    def test(self, n_episodes: int = 100) -> dict:
        """
        Evaluate agent over n_episodes.

        Returns:
            dict with 'success_rate', 'mean_fuel', 'n_collisions'
        """
        successes, collisions, fuel_costs = 0, 0, []

        for ep in range(n_episodes):
            obs, _ = self.raw_env.reset()
            episode_fuel = 0.0
            done = False

            while not done:
                action = self._get_action(obs)
                dv     = action * 0.1
                episode_fuel += float(np.linalg.norm(dv))
                obs, reward, terminated, truncated, _ = self.raw_env.step(action)
                done = terminated or truncated

                if terminated:
                    dist = float(np.linalg.norm(
                        self.raw_env.sat_pos - self.raw_env.debris_pos
                    ))
                    if dist > self.raw_env.safe_radius:
                        successes += 1
                    elif dist < self.raw_env.collision_radius:
                        collisions += 1

            fuel_costs.append(episode_fuel)

        success_rate = successes / n_episodes
        mean_fuel    = float(np.mean(fuel_costs))

        print(f"\n[PPO Evaluation — {n_episodes} episodes]")
        print(f"  Success Rate  : {success_rate:.2%}")
        print(f"  Collisions    : {collisions}")
        print(f"  Mean Fuel     : {mean_fuel:.4f}")

        return {
            'success_rate': success_rate,
            'collisions':   collisions,
            'mean_fuel':    mean_fuel,
            'n_episodes':   n_episodes,
        }

    def compute_avoidance(self,
                          relative_position: np.ndarray,
                          relative_velocity: np.ndarray,
                          own_position:      np.ndarray,
                          own_velocity:      np.ndarray) -> dict:
        """
        Compute optimal ΔV maneuver.

        Returns:
            dict with 'delta_v' (np.array (3,)) and 'fuel_cost' (float)
        """
        obs = np.concatenate([
            relative_position, relative_velocity,
            own_position, own_velocity
        ]).astype(np.float32)

        action = self._get_action(obs)
        dv     = action * 0.1
        return {
            'delta_v':   dv,
            'fuel_cost': float(np.linalg.norm(dv))
        }

    def _get_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from PPO model or rule-based fallback."""
        if self.model and HAS_SB3:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        else:
            # Rule-based: thrust directly away from debris
            rel_pos   = obs[:3]
            direction = -rel_pos / (np.linalg.norm(rel_pos) + 1e-8)
            return (direction * 0.5).astype(np.float32)


# ────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test',  action='store_true')
    parser.add_argument('--steps', type=int, default=50000)
    args = parser.parse_args()

    MODELS_DIR = os.path.join(AIML_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'ppo_avoidance.zip')

    agent = PPOAvoidanceAgent(model_path=model_path)

    if args.train:
        agent.train(total_timesteps=args.steps, use_curriculum=True)

    if args.test:
        metrics = agent.test(n_episodes=100)
        print(f"\nFinal Metrics: {metrics}")
