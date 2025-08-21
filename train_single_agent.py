from utils.episode_saver import EpisodeSaver
from utils.episode_train_policy import TrainingEvaluator
import utils.warmup
from utils import save

import torch
import numpy as np
from env.drone_env import DroneEnv
from agents.ddpg_agent import DDPGAgent
from utils.generation import EnvGeneratorDifferentEpisodes, EnvGeneratorDynamic
from utils.device_support import get_device
from utils.db import DBSaver
import random

from config.env_config import FIELD_SIZE
from config.train_config import (
    NUM_EPISODES,
    MAX_STEPS_PER_EPISODE,
    TRAIN_COUNT,
    TEST_COUNT,
    EVAL_INTERVAL,
    ACTOR_LR,
    CRITIC_LR,
    GAMMA,
    TAU,
    BUFFER_SIZE,
    BATCH_SIZE,
    ACTION_NOISE_STD,
    ACTION_NOISE_STD2, ACTION_NOISE_STD3,
)
from utils.save import plot_episode_summary

DEVICE = get_device()

def train():
    db_saver = DBSaver()

    env_train_generator = EnvGeneratorDynamic(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    env_test_generator = EnvGeneratorDifferentEpisodes(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    train_evaluator = TrainingEvaluator(TRAIN_COUNT, TEST_COUNT)
    env = DroneEnv(env_test_generator)
    obs_dim = env.observation_space.shape[0]
    action_dim = 2  # [dx, dy] Actor output

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

    episode_saver = EpisodeSaver()

    utils.warmup.generate_warmup_experience(agent, FIELD_SIZE)

    rewards_history = []
    replay_buffer_mean_history = []

    for episode in range(NUM_EPISODES):
        is_train = train_evaluator.get_policy()
        db_saver.start_new_episode(episode, is_train)

        if is_train:
            env.env_generator = env_train_generator
        else:
            env.env_generator = env_test_generator
            episode_saver.start_episode(env)

        total_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            env.env_generator.set_state(episode=episode, step=step)
            obs, _ = env.reset()
        
            noise_std = get_noise(
                episode=episode, num_episodes=NUM_EPISODES,
                step=step, num_steps=MAX_STEPS_PER_EPISODE
            ) if is_train else 0.0

            

            move_vector = agent.select_action(obs, noise_std=noise_std)

            action_for_buffer = move_vector.detach().clone()

            previous_drone_pos = env.env_data.drone.position.copy()
            
            next_obs, reward, terminated, truncated, details = env.step(move_vector)
            done = terminated or truncated

            db_saver.add_step(step,
                previous_drone_pos[0], previous_drone_pos[1],
                env.env_data.drone.position[0], env.env_data.drone.position[1],
                action_for_buffer[0], action_for_buffer[1],
                details['last_distance'], details['new_distance'],
                reward, details['result'],
            )

            if is_train:
                agent.replay_buffer.add(obs, action_for_buffer.cpu().numpy(), reward, next_obs, float(done))
                agent.update()

                replay_buffer_mean_history.append(agent.replay_buffer.get_stats())
            else:
                episode_saver.add_rewards(
                    details['target_reward'],
                    details['obstacle_penalty'],
                    details['step_penalty']
                )
                episode_saver.add_drone_pos(env.env_data.drone.position.copy())
            obs = next_obs
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)

        if not is_train:
            plot_episode_summary(episode=episode, saver=episode_saver)
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode+1}, Average Reward: {avg_reward:.2f}")

        if (episode + 1) % EVAL_INTERVAL == 0:
            save.save_checkpoint(agent, episode)
        
        train_evaluator.update()

    save.save_critic_loss(agent)
    save.save_actor_loss(agent)
    save.save_rewards(rewards_history)
    save.save_replay_buffer(replay_buffer_mean_history)
    save.plot_td_histogram(agent.replay_buffer.priorities)


def get_noise(episode: int, num_episodes: int, step: int, num_steps: int):
    if episode / num_episodes >= 0.4 or episode > 600:
        return ACTION_NOISE_STD2
    elif episode / num_episodes >= 0.8 or episode > 1500:
        return ACTION_NOISE_STD3
    else:
        return ACTION_NOISE_STD


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    save.create_save_dir()
    train()
