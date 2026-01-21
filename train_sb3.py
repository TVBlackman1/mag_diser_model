"""
Training script using Stable-Baselines3 instead of custom implementation.
This replaces the custom DDPG agent with SB3's DDPG/TD3 implementation.
"""
import argparse
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
from config.env_config import (
    FIELD_SIZE,
    TARGET_MAX_SPEED,
    TARGET_MAX_ANGULAR_VELOCITY,
    TARGET_MIN_DISTANCE_TO_OBSTACLE,
)
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
from utils.target_strategies import StaticTargetStrategy, RandomMovingTargetStrategy, FleeingTargetStrategy, TargetStrategy
from config.version import EXPERIMENT_FOLDER

DEVICE = get_device()

TARGET_STRATEGIES = ("static", "random", "fleeing")


def _make_target_strategy(name: str, seed: int = 0) -> TargetStrategy | None:
    rng = np.random.default_rng(seed)
    if name == "static":
        return StaticTargetStrategy(rng=rng)
    if name == "random":
        return RandomMovingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=rng,
        )
    if name == "fleeing":
        return FleeingTargetStrategy(
            max_speed=TARGET_MAX_SPEED,
            max_angular_velocity=TARGET_MAX_ANGULAR_VELOCITY,
            min_distance_to_obstacle=TARGET_MIN_DISTANCE_TO_OBSTACLE,
            rng=rng,
        )
    raise ValueError(f"Unknown target strategy: {name}")


def create_env(is_eval: bool = False, db_saver=None, target_strategy: str = "static"):
    """Create environment instance with optional DB logging."""
    if is_eval:
        generator = EnvGeneratorDifferentEpisodes(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    else:
        generator = EnvGeneratorDynamic(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    
    strategy = _make_target_strategy(target_strategy, seed=0) if target_strategy else None
    env = DroneEnv(generator, target_strategy=strategy)
    
    # Wrap with state wrapper to set generator state automatically.
    # Important: do NOT force 'medium' difficulty here; let generator decide (or pass explicitly).
    env = DroneEnvStateWrapper(env, difficulty_level="")
    
    # Wrap with DB logging if db_saver is provided
    if db_saver is not None:
        env = DroneEnvDBWrapper(env, db_saver)
    
    # Wrap with Monitor for SB3 logging (must be last)
    env = Monitor(env)
    return env


def _train_one(target_strategy: str, *, use_db: bool = True):
    """Train a separate model for a single target behavior."""
    run_folder = f"{EXPERIMENT_FOLDER}/{target_strategy}"
    create_save_dir(base_folder=run_folder)
    db_saver = DBSaver(base_folder=run_folder) if use_db else None
    
    # Create environments with DB logging
    train_env = create_env(is_eval=False, db_saver=db_saver, target_strategy=target_strategy)
    eval_env = create_env(is_eval=True, db_saver=db_saver, target_strategy=target_strategy)
    
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
        tensorboard_log=f"./tensorboard_logs/{target_strategy}/",
        verbose=1,
        device=DEVICE,
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=EVAL_INTERVAL * MAX_STEPS_PER_EPISODE,
        save_path=f"./checkpoints/{target_strategy}/",
        name_prefix=f"td3_{target_strategy}",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./best_model/{target_strategy}/",
        log_path=f"./eval_logs/{target_strategy}/",
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
    model.save(f"./checkpoints/{target_strategy}/td3_{target_strategy}_final")
    print(f"Training completed for target strategy: {target_strategy}")


def train():
    """Train models for one or all target behaviors in a single run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-strategy", type=str, default="all", choices=(*TARGET_STRATEGIES, "all"))
    ap.add_argument("--no-db", action="store_true", help="Disable DuckDB logging during training")
    args = ap.parse_args()

    if args.target_strategy == "all":
        for s in TARGET_STRATEGIES:
            _train_one(s, use_db=not args.no_db)
    else:
        _train_one(args.target_strategy, use_db=not args.no_db)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    train()
