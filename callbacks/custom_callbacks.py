"""
Custom callbacks for Stable-Baselines3 training.
These callbacks preserve custom functionality like DB logging and visualization.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional

from utils.db import DBSaver
from utils.episode_saver import EpisodeSaverStatic
from utils.save import plot_episode_summary
from utils.episode_train_policy import TrainingEvaluator
from utils.generation import EnvGenerator


# DBCallback removed - DB logging is now handled by DroneEnvDBWrapper


class EpisodeVisualizationCallback(BaseCallback):
    """Callback for saving episode visualizations during evaluation."""
    
    def __init__(self, episode_saver: EpisodeSaverStatic, save_freq: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.episode_saver = episode_saver
        self.save_freq = save_freq
        self.eval_episode_count = 0
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Save visualization if in evaluation mode."""
        # This would need to be called during evaluation episodes
        pass


class TrainingPolicyCallback(BaseCallback):
    """Callback to switch between training and evaluation modes."""
    
    def __init__(
        self, 
        train_evaluator: TrainingEvaluator,
        env_train_generator: EnvGenerator,
        env_test_generator: EnvGenerator,
        env,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.train_evaluator = train_evaluator
        self.env_train_generator = env_train_generator
        self.env_test_generator = env_test_generator
        self.env = env
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # SB3 off-policy uses rollouts that do NOT equal episodes.
        # Count episodes via `done` signals.
        dones = self.locals.get("dones")
        if dones is not None and bool(np.any(dones)):
            self.episode_count += 1
            # Advance mode AFTER finishing episode so the next episode uses the new mode.
            self.train_evaluator.update()
            is_train = self.train_evaluator.get_policy()

            # Update environment generator on the base env for the next episode
            if hasattr(self.env, "env_generator"):
                self.env.env_generator = self.env_train_generator if is_train else self.env_test_generator

        return True
    
    def _on_rollout_end(self) -> None:
        """Switch environment generator based on training/evaluation mode."""
        # Kept for compatibility, but episode switching is handled in _on_step via `done`.
        return


class AdaptiveNoiseCallback(BaseCallback):
    """Callback to adaptively change action noise during training."""
    
    def __init__(
        self,
        model,
        initial_noise_std: float = 0.5,
        final_noise_std: float = 0.03,
        noise_decay_episodes: int = 1500,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.model = model
        self.initial_noise_std = initial_noise_std
        self.final_noise_std = final_noise_std
        self.noise_decay_episodes = noise_decay_episodes
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        """Called after each step."""
        dones = self.locals.get("dones")
        if dones is not None and bool(np.any(dones)):
            self.episode_count += 1

            # Calculate current noise std
            if self.episode_count < self.noise_decay_episodes:
                progress = self.episode_count / self.noise_decay_episodes
                current_noise_std = self.initial_noise_std * (1 - progress) + self.final_noise_std * progress
            else:
                current_noise_std = self.final_noise_std

            # Update model's action noise if it has one
            if hasattr(self.model, 'action_noise') and self.model.action_noise is not None:
                if hasattr(self.model.action_noise, 'sigma'):
                    n_actions = len(self.model.action_noise.sigma)
                    self.model.action_noise.sigma = current_noise_std * np.ones(n_actions)

            if self.verbose > 0 and self.episode_count % 100 == 0:
                print(f"Episode {self.episode_count}: Action noise std = {current_noise_std:.4f}")

        return True
        
    def _on_rollout_end(self) -> None:
        """Update action noise based on training progress."""
        # Kept for compatibility; noise schedule is applied per-episode in _on_step via `done`.
        return
