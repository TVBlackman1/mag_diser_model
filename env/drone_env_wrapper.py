"""
Wrapper for DroneEnv that adds DB logging functionality for use with Stable-Baselines3.
"""
import numpy as np
from gymnasium import Wrapper
from typing import Any

from utils.db import DBSaver
from env.drone_env import DroneEnv


class DroneEnvDBWrapper(Wrapper):
    """Wrapper that logs all steps to DuckDB database."""
    
    def __init__(self, env: DroneEnv, db_saver: DBSaver):
        super().__init__(env)
        self.db_saver = db_saver
        self.step_count = 0
        # Use sentinel so episode 0 is also initialized on first reset().
        self.episode_count = -1
        self.is_train = True
        
    def reset(self, *, seed=None, options=None):
        """Reset environment and start new episode in DB."""
        obs, info = self.env.reset(seed=seed, options=options)
        self.step_count = 0
        
        # Get the base DroneEnv (unwrap if needed)
        base_env = self.env
        while hasattr(base_env, 'env'):
            # Stop at DroneEnv
            if isinstance(base_env.env, DroneEnv):
                base_env = base_env.env
                break
            base_env = base_env.env
        
        # Check if this is a new episode
        if isinstance(base_env, DroneEnv) and hasattr(base_env, 'env_generator') and base_env.env_generator is not None:
            current_episode = base_env.env_generator.episode
            if current_episode != self.episode_count:
                self.episode_count = current_episode
                # Determine if training or evaluation based on generator type
                from utils.generation import EnvGeneratorDynamic, EnvGeneratorDifferentEpisodes
                self.is_train = isinstance(base_env.env_generator, EnvGeneratorDynamic)
                self.db_saver.start_new_episode(self.episode_count, self.is_train)
        
        return obs, info
    
    def step(self, action):
        """Step environment and log to DB."""
        # Get the base DroneEnv (unwrap if needed)
        base_env = self.env
        while hasattr(base_env, 'env'):
            # Stop at DroneEnv
            if isinstance(base_env.env, DroneEnv):
                base_env = base_env.env
                break
            base_env = base_env.env
        
        # Store previous state
        if isinstance(base_env, DroneEnv) and hasattr(base_env, 'env_data') and base_env.env_data is not None:
            previous_drone_pos = base_env.env_data.drone.position.copy()
        else:
            previous_drone_pos = np.array([0.0, 0.0])
        
        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Log to DB if we have environment data
        if isinstance(base_env, DroneEnv) and hasattr(base_env, 'env_data') and base_env.env_data is not None:
            current_drone_pos = base_env.env_data.drone.position.copy()
            
            # Extract action values
            if isinstance(action, np.ndarray):
                speed_ratio = float(action[0])
                angle_ratio = float(action[1])
            else:
                speed_ratio = float(action[0])
                angle_ratio = float(action[1])
            
            # Extract distances and result from info
            last_distance = info.get('last_distance', -1.0)
            new_distance = info.get('new_distance', -1.0)
            result = info.get('result', 'process')
            
            # Log step to DB
            self.db_saver.add_step(
                step=self.step_count,
                x=previous_drone_pos[0],
                y=previous_drone_pos[1],
                new_x=current_drone_pos[0],
                new_y=current_drone_pos[1],
                speed_ratio=speed_ratio,
                angle_ratio=angle_ratio,
                target_distance=last_distance,
                new_target_distance=new_distance,
                reward=float(reward),
                result=result
            )
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info
