import csv
import os
from typing import List

import numpy as np
from array import ArrayType
from matplotlib import pyplot as plt

from config.version import EXPERIMENT_FOLDER, EXPERIMENT_NOTES

CHECKPOINTS_SUBDIR = "checkpoints"
CHECKPOINTS_DIR = f"{EXPERIMENT_FOLDER}/{CHECKPOINTS_SUBDIR}"
LOG_BUFFER_LIMIT = 15

def create_save_dir():
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    with open(f'{EXPERIMENT_FOLDER}/description.txt', 'w', newline='') as file:
        file.write(EXPERIMENT_NOTES)

def save_checkpoint(agent, episode):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    agent.save(f"{CHECKPOINTS_DIR}/ddpg_agent_episode_{episode+1:04d}.pth")

class CSVSaver:
    def __init__(self, filename):
        self._csv_file = open(f"{EXPERIMENT_FOLDER}/{filename}.csv", mode='w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._buffer = []

    def write(self, data: List):
        self._buffer.append(data)
        if len(self._buffer) >= LOG_BUFFER_LIMIT:
            self._flush()

    def close(self):
        self._flush()
        self._csv_file.close()

    def _flush(self):
        self._csv_writer.writerows(self._buffer)
        self._csv_file.flush()
        self._buffer = []

def save_critic_loss(agent):
    plt.figure(figsize=(10, 6))
    plt.plot(agent.critic_loss_history, label='Critic Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.title('Critic Loss over Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{EXPERIMENT_FOLDER}/critic_loss_plot.png"
    plt.savefig(filename)

def save_actor_loss(agent):
    plt.figure(figsize=(10, 6))
    plt.plot(agent.actor_loss_history, label='Actor Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('MSE Loss')
    plt.title('Actor Loss over Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{EXPERIMENT_FOLDER}/actor_loss_plot.png"
    plt.savefig(filename)

def save_rewards(rewards, avg_window=10):
    episodes = np.arange(len(rewards))
    rewards = np.array(rewards)

    # Скользящее среднее для сглаживания
    avg_rewards = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Episode Reward")
    plt.plot(episodes[avg_window-1:], avg_rewards, label=f"Moving Average ({avg_window} eps)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)

    filename = f"{EXPERIMENT_FOLDER}/rewards_plot.png"
    plt.savefig(filename)

def save_replay_buffer(buffer: List):
    plt.figure(figsize=(10, 6))
    plt.plot(buffer, label='Replay buffer reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Mean reward')
    plt.title('Replay Buffer mean over Training')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"{EXPERIMENT_FOLDER}/replay_buffer_mean_reward_plot.png"
    plt.savefig(filename)

