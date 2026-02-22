import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from avoidance.rl_agent import SpaceAvoidanceEnv

def train_rl_agent():
    print("🚀 Initializing Deep Reinforcement Learning (PPO) Training...")
    
    # Create logs directory
    log_dir = os.path.join(BASE_DIR, "models", "rl_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrap environment with Monitor correctly recording episodic rewards
    env = Monitor(SpaceAvoidanceEnv(), log_dir)
    
    # Define Agent
    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=log_dir)
    
    # Hyperparameters for demo
    timesteps = 25000
    print(f"🧠 Training PPO Agent for {timesteps} interactions...")
    
    # Execute Training
    model.learn(total_timesteps=timesteps, progress_bar=True)
    
    # Save Weights
    model_path = os.path.join(BASE_DIR, "models", "ppo_avoidance.zip")
    model.save(model_path)
    print(f"✅ RL Policy saved securely to {model_path}")
    
    # Generate Training Curves
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, alpha=0.3, c='blue', s=8, label="Episode Reward")
        
        # Add smooth moving average
        y_smooth = pd.Series(y).rolling(window=max(2, len(y)//20)).mean()
        plt.plot(x, y_smooth, c='red', lw=2, label="Moving Average")
        
        plt.xlabel('Timesteps Simulated')
        plt.ylabel('Episodic Reward')
        plt.title('PPO Collision Avoidance Learning Curve')
        plt.legend()
        plt.grid()
        
        plot_path = os.path.join(BASE_DIR, "models", "ppo_learning_curve.png")
        plt.savefig(plot_path)
        print(f"📈 Real RL Training curve exported to {plot_path}")
    except Exception as e:
        print(f"⚠️ Could not generate reward plot: {e}")

if __name__ == "__main__":
    train_rl_agent()
