"""
Wrapper that sets generator state before reset for compatibility with SB3.
SB3 doesn't call set_state explicitly, so we need to track timesteps and set state automatically.
"""
from gymnasium import Wrapper
from typing import Any

from env.drone_env import DroneEnv


class DroneEnvStateWrapper(Wrapper):
    """Wrapper that automatically sets generator state before reset."""
    
    def __init__(self, env: DroneEnv, difficulty_level: str = ""):
        super().__init__(env)
        self.total_timesteps = 0
        self.current_episode = 0
        self.current_step = 0
        # If empty, generators may choose difficulty automatically (recommended).
        self.difficulty_level = difficulty_level
        
    def reset(self, *, seed=None, options=None):
        """Reset environment and update generator state."""
        # Gymnasium semantics: reset() starts a new episode.
        self.current_step = 0

        # Set generator state before reset (so generation can be deterministic)
        if hasattr(self.env, 'env_generator') and self.env.env_generator is not None:
            self.env.env_generator.set_state(
                episode=self.current_episode,
                step=self.current_step,
                difficult_level=self.difficulty_level
            )
        
        obs, info = self.env.reset(seed=seed, options=options)

        return obs, info
    
    def step(self, action):
        """Step environment and update timestep counter."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.total_timesteps += 1
        self.current_step += 1
        
        # If episode ended, advance episode counter so next reset() sets correct generator state.
        if terminated or truncated:
            self.current_episode += 1
            self.current_step = 0
        
        return obs, reward, terminated, truncated, info
