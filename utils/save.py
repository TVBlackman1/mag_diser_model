import csv
import os
from typing import List
import torch
import numpy as np
from matplotlib import pyplot as plt

from config.version import EXPERIMENT_FOLDER, EXPERIMENT_NOTES
from env.drone_env import DroneEnv

CHECKPOINTS_SUBDIR = "checkpoints"
Q_SURFACE_SUBDIR = "q_surfaces"
CHECKPOINTS_DIR = f"{EXPERIMENT_FOLDER}/{CHECKPOINTS_SUBDIR}"
Q_SURFACE_DIR = f"{EXPERIMENT_FOLDER}/{Q_SURFACE_SUBDIR}"
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

def save_replay_buffer(stats: List):
    means = [s["mean"] for s in stats]
    maxs = [s["max"] for s in stats]
    mins = [s["min"] for s in stats]
    medians = [s["median"] for s in stats]

    episodes = list(range(1, len(stats) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, means, label="Mean TD", color="blue")
    plt.plot(episodes, medians, label="Median TD", color="purple", linestyle="-.")
    plt.plot(episodes, maxs, label="Max TD", color="red", linestyle="--")
    plt.plot(episodes, mins, label="Min TD", color="green", linestyle="--")

    plt.title("TD Priority Progression per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Priority (TD Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{EXPERIMENT_FOLDER}/replay_buffer_mean_reward_plot.png"
    plt.savefig(filename)


def plot_td_histogram(td_errors, bins=50):
    td_errors = np.array(td_errors)

    plt.figure(figsize=(10, 5))
    plt.hist(td_errors, bins=bins, color='skyblue', edgecolor='black')
    plt.title("Histogram of Final TD Errors in Replay Buffer")
    plt.xlabel("TD Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    filename = f"{EXPERIMENT_FOLDER}/replay_buffer_td_histogram_plot.png"
    plt.savefig(filename)

def save_q_surface(critic, env: DroneEnv, episode, resolution=50, stride=4):
    os.makedirs(Q_SURFACE_DIR, exist_ok=True)

    obs_tensor = torch.tensor(env.get_current_obs(), dtype=torch.float32).unsqueeze(0)

    dx = np.linspace(-1, 1, resolution)
    dy = np.linspace(-1, 1, resolution)
    values = np.zeros((resolution, resolution))

    for i, x in enumerate(dx):
        for j, y in enumerate(dy):
            action = torch.tensor([[x, y]], dtype=torch.float32)
            q_val = critic(obs_tensor, action).item()
            values[j, i] = q_val

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Q-функция в пространстве действий
    im = ax1.imshow(values, extent=[-1, 1, -1, 1], origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax1, label='Q value')
    ax1.set_title("Q(s_fixed, a)")
    ax1.set_xlabel("dx")
    ax1.set_ylabel("dy")

    # --- Мир (относительные координаты)
    ax2.set_title("World space (centered on drone)")
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.plot(0, 0, 'bo', label="Drone")
    ax3.plot(0, 0, 'bo', label="Drone")

    max_dist = 1.0
    if env.target_pos is not None:
        relative_target = np.array(env.target_pos) - env.drone_pos
        ax2.plot(relative_target[0], relative_target[1], 'ro', label="Target")
        ax3.plot(relative_target[0], relative_target[1], 'ro', label="Target")
        max_dist = max(max_dist, float(np.linalg.norm(relative_target)))

    if env.obstacles:
        for i, obs in enumerate(env.obstacles):
            rel_obs = np.array(obs) - env.drone_pos
            ax2.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)
            ax3.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)
            max_dist = max(max_dist, np.linalg.norm(rel_obs))

    lim = max_dist * 1.2
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.legend()

    # --- Фазовый портрет (в каждой точке мира: куда идти)
    ax3.set_title("Phase portrait in world space")
    ax3.set_aspect('equal')
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.grid(True)

    portrait_resolution = 24
    world_grid = np.linspace(-lim, lim, portrait_resolution)
    X, Y = np.meshgrid(world_grid, world_grid)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    action_dx = np.linspace(-1, 1, 8)
    action_dy = np.linspace(-1, 1, 8)

    for i in range(portrait_resolution):
        for j in range(portrait_resolution):
            pos = np.array([X[i, j], Y[i, j]]) + env.drone_pos
            tmp_env = DroneEnv()
            tmp_env.set_positions(pos, env.target_pos, env.obstacles)
            obs = tmp_env.get_current_obs()
            obs_tensor_local = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            best_q = -float('inf')
            best_a = (0.0, 0.0)

            for dx_ in action_dx:
                for dy_ in action_dy:
                    a = torch.tensor([[dx_ , dy_]], dtype=torch.float32)
                    q = critic(obs_tensor_local, a).item()
                    if q > best_q:
                        best_q = q
                        best_a = (dx_, dy_)

            norm = float(np.linalg.norm(best_a))
            if norm > 1e-5:
                U[i, j] = best_a[0] / norm
                V[i, j] = best_a[1] / norm
            else:
                U[i, j] = 0.0
                V[i, j] = 0.0

    ax3.quiver(X, Y, U, V, color='black', scale=20, alpha=0.6)

    # Сохраняем
    plt.tight_layout()
    filename = f"{Q_SURFACE_DIR}/q_surface_{episode+1:04d}.png"
    plt.savefig(filename)
    plt.close()