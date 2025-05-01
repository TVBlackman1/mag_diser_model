import utils.warmup
from utils import save

import torch
import numpy as np
from env.drone_env import DroneEnv
from agents.ddpg_agent import DDPGAgent
from agents.ddpg_agent import DIRECTIONS
from utils import time_logger
import random

from config.train_config import (
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    EVAL_INTERVAL,
    ACTOR_LR,
    CRITIC_LR,
    GAMMA,
    TAU,
    BUFFER_SIZE,
    BATCH_SIZE,
    ACTION_NOISE_STD,
    ACTION_NOISE_STD2,
    ACTION_NOISE_STD3,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    csv_log = save.CSVSaver("training_log")
    csv_log.write(["Episode", "InitialDistance", "FinalDistance", "PercentCovered", "Reward"])

    env = DroneEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = 2  # [dx, dy] выход Actor-а

    agent = DDPGAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    utils.warmup.generate_warmup_experience(agent)

    rewards_history = []
    replay_buffer_mean_history = []

    for episode in range(NUM_EPISODES):
        time_logger.start("env.reset")
        obs, _ = env.reset()
        time_logger.stop("env.reset")
        initial_distance = obs[3] * env.max_distance
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            noise_std = ACTION_NOISE_STD
            if step >= 200:
                noise_std = ACTION_NOISE_STD2

            time_logger.start("action")
            action_idx = agent.select_action(obs, noise_std=noise_std)
            time_logger.stop("action")

            move_direction = DIRECTIONS[action_idx]
            action_for_buffer = torch.tensor(move_direction, dtype=torch.float32).to(DEVICE)

            time_logger.start("env.step")
            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            time_logger.stop("env.step")

            done = terminated or truncated

            time_logger.start("replay_buffer.add")
            agent.replay_buffer.add(obs, action_for_buffer.cpu().numpy(), reward, next_obs, float(done))
            time_logger.stop("replay_buffer.add")

            replay_buffer_mean_history.append(agent.replay_buffer.mean)

            time_logger.start("agent.update")
            agent.update()
            time_logger.stop("agent.update")


            obs = next_obs
            total_reward += reward

            if done:
                break
        final_distance = obs[3] * env.max_distance
        distance_traveled = initial_distance - final_distance
        percent_path_covered = (distance_traveled / initial_distance) * 100 if initial_distance > 1e-5 else 0.0

        csv_log.write([
            episode + 1,
            round(initial_distance, 4),
            round(final_distance, 4),
            round(percent_path_covered, 2),
            total_reward,
        ])

        rewards_history.append(total_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")

        if (episode + 1) % EVAL_INTERVAL == 0:
            save.save_checkpoint(agent, episode)

    csv_log.close()
    save.save_critic_loss(agent)
    save.save_rewards(rewards_history)
    save.save_replay_buffer(replay_buffer_mean_history)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    save.create_save_dir()
    train()
    time_logger.export_csv()
