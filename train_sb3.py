"""
Training script using Stable-Baselines3 instead of custom implementation.
This replaces the custom DDPG agent with SB3's DDPG/TD3 implementation.
"""
import random
import numpy as np
import torch
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor

from env.drone_env import DroneEnv
from env.drone_env_wrapper import DroneEnvDBWrapper
from env.drone_env_state_wrapper import DroneEnvStateWrapper
from utils.generation import EnvGeneratorDynamic, EnvGeneratorDifferentEpisodes
from utils.db import DBSaver
from utils.save import create_save_dir
from config.env_config import FIELD_SIZE
from config.train_config import (
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    EVAL_INTERVAL,
    ACTOR_LR,
    GAMMA,
    TAU,
    BUFFER_SIZE,
    BATCH_SIZE,
    ACTION_NOISE_STD,
)
from utils.device_support import get_device
from callbacks import AdaptiveNoiseCallback
from stable_baselines3.common.monitor import Monitor

DEVICE = get_device()


def create_env(is_eval=False, db_saver=None):
    """Create environment instance with optional DB logging."""
    if is_eval:
        generator = EnvGeneratorDifferentEpisodes(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    else:
        generator = EnvGeneratorDynamic(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    
    env = DroneEnv(generator)
    
    # Wrap with state wrapper to set generator state automatically.
    # Important: do NOT force 'medium' difficulty here; let generator decide (or pass explicitly).
    env = DroneEnvStateWrapper(env, difficulty_level="")
    
    # Wrap with DB logging if db_saver is provided
    if db_saver is not None:
        env = DroneEnvDBWrapper(env, db_saver)
    
    # Wrap with Monitor for SB3 logging (must be last)
    env = Monitor(env)
    return env


def train():
    """Main training function using Stable-Baselines3."""
    create_save_dir()
    db_saver = DBSaver()
    
    # Create environments with DB logging
    train_env = create_env(is_eval=False, db_saver=db_saver)
    eval_env = create_env(is_eval=True, db_saver=db_saver)
    
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    # Create action noise - Ornstein-Uhlenbeck is traditional for DDPG
    n_actions = action_dim
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        sigma=ACTION_NOISE_STD * np.ones(n_actions),
        theta=0.15,
        dt=0.1
    )
    
    # Create model - using TD3 as it's more stable than DDPG
    # Can switch to DDPG by changing TD3 to DDPG
    # Note: TD3 uses learning_rate for both actor and critic (can be overridden)
    model = TD3(
        "MlpPolicy",
        train_env,
        learning_rate=ACTOR_LR,  # Base learning rate
        buffer_size=BUFFER_SIZE,
        learning_starts=1000,  # Warm-up period (collect experience before training)
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        action_noise=action_noise,
        policy_kwargs=dict(
            net_arch=[256, 256],  # Match your original architecture
        ),
        tensorboard_log="./tensorboard_logs/",
        verbose=1,
        device=DEVICE,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_INTERVAL * MAX_STEPS_PER_EPISODE,
        save_path="./checkpoints/",
        name_prefix="td3_drone",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=EVAL_INTERVAL * MAX_STEPS_PER_EPISODE,
        deterministic=True,
        render=False,
    )
    
    # Adaptive noise callback
    adaptive_noise_callback = AdaptiveNoiseCallback(
        model,
        initial_noise_std=ACTION_NOISE_STD,
        final_noise_std=0.03,
        noise_decay_episodes=1500,
    )
    
    # Train the model
    total_timesteps = NUM_EPISODES * MAX_STEPS_PER_EPISODE
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            checkpoint_callback, 
            eval_callback, 
            adaptive_noise_callback,
        ],
        log_interval=10,
        progress_bar=True,
    )
    
    # Save final model
    model.save(f"./checkpoints/td3_drone_final")
    print("Training completed!")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    train()
