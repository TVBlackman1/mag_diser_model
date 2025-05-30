import csv
import os
from typing import List
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

from agents.ddpg_agent import DDPGAgent
from config.version import EXPERIMENT_FOLDER, EXPERIMENT_NOTES
from env.drone_env import DroneEnv
from utils.episode_saver import EpisodeSaver

CHECKPOINTS_SUBDIR = "checkpoints"
Q_SURFACE_SUBDIR = "q_surfaces"
PATHS_SUBDIR = "paths"
CHECKPOINTS_DIR = f"{EXPERIMENT_FOLDER}/{CHECKPOINTS_SUBDIR}"
Q_SURFACE_DIR = f"{EXPERIMENT_FOLDER}/{Q_SURFACE_SUBDIR}"
PATHS_DIR = f"{EXPERIMENT_FOLDER}/{PATHS_SUBDIR}"
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

def save_q_surface(agent: DDPGAgent, env: DroneEnv, episode, resolution=50):
    os.makedirs(Q_SURFACE_DIR, exist_ok=True)

    critic = agent.critic
    actor = agent.actor

    obs_tensor = torch.tensor(env.get_current_obs(), dtype=torch.float32).unsqueeze(0)

    dx = np.linspace(-1, 1, resolution)
    dy = np.linspace(-1, 1, resolution)
    values = np.zeros((resolution, resolution))

    for i, x in enumerate(dx):
        for j, y in enumerate(dy):
            action = torch.tensor([[x, y]], dtype=torch.float32)
            q_val = critic(obs_tensor, action).item()
            values[j, i] = q_val

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]

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
    ax4.plot(0, 0, 'bo', label="Drone")

    max_dist = 1.0
    if env.target_pos is not None:
        relative_target = np.array(env.target_pos) - env.drone_pos
        ax2.plot(relative_target[0], relative_target[1], 'ro', label="Target")
        ax3.plot(relative_target[0], relative_target[1], 'ro', label="Target")
        ax4.plot(relative_target[0], relative_target[1], 'ro', label="Target")
        max_dist = max(max_dist, float(np.linalg.norm(relative_target)))

    if env.obstacles:
        for i, obs in enumerate(env.obstacles):
            rel_obs = np.array(obs) - env.drone_pos
            ax2.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)
            ax3.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)
            ax4.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)
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

    # --- Фазовый портрет от актора
    ax4.set_title("Actor Policy Field")
    ax4.set_aspect('equal')
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.grid(True)

    portrait_resolution = 24
    world_grid = np.linspace(-lim, lim, portrait_resolution)
    X, Y = np.meshgrid(world_grid, world_grid)
    U_critic = np.zeros_like(X)
    U_actor = np.zeros_like(X)
    V_critic = np.zeros_like(Y)
    V_actor = np.zeros_like(Y)

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

            with torch.no_grad():
                action = actor(obs_tensor_local).squeeze().numpy()

            U_critic[i, j], V_critic[i, j] = norm_vector(best_a)
            U_actor[i, j], V_actor[i, j] = norm_vector(action)

    ax3.quiver(X, Y, U_critic, V_critic, color='black', scale=20, alpha=0.6)
    ax4.quiver(X, Y, U_actor, V_actor, color='darkgreen', scale=20, alpha=0.6)
    ax4.set_xlim(-lim, lim)
    ax4.set_ylim(-lim, lim)

    # Сохраняем
    plt.tight_layout()
    filename = f"{Q_SURFACE_DIR}/q_surface_{episode+1:04d}.png"
    plt.savefig(filename)
    plt.close()

def norm_vector(vector):
    norm = float(np.linalg.norm(vector))
    if norm > 1e-5:
       return vector[0] / norm, vector[1] / norm
    return 0.0, 0.0



def plot_episode_summary(episode, saver: EpisodeSaver):
    os.makedirs(PATHS_DIR, exist_ok=True)

    # fig, (ax_map, ax_hist) = plt.subplots(2, 1, figsize=(10, 14), gridspec_kw={'height_ratios': [2, 1]})
    fig, (ax_map, ax_hist) = plt.subplots(
        2, 1,
        figsize=(30, 14),  # ширина *3, высота та же
        gridspec_kw={'height_ratios': [2, 1]}
    )
    # === Карта ===
    ax = ax_map
    ax.set_title("World space (centered on drone)")
    ax.set_aspect('equal')
    ax.grid(True)

    drone_center = saver.drone_poses[0]
    ax.plot(0, 0, 'bo', label="Drone (start)")

    max_dist = 1.0

    # Цель
    if saver.target_pos is not None:
        relative_target = np.array(saver.target_pos) - drone_center
        max_dist = max(max_dist, float(np.linalg.norm(relative_target)))
        ax.plot(relative_target[0], relative_target[1], 'ro', label="Target")

    # Препятствия
    if saver.obstacles:
        for i, obs in enumerate(saver.obstacles):
            rel_obs = np.array(obs) - drone_center
            max_dist = max(max_dist, np.linalg.norm(rel_obs))
            ax.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)

    # Путь дрона
    rel_poses = [np.array(pos) - drone_center for pos in saver.drone_poses]
    rel_poses = np.array(rel_poses)
    num_points = len(rel_poses)
    cmap_steps = cm.get_cmap('tab20', num_points)

    for i in range(num_points):
        step_color = cmap_steps(i)
        ax.plot(rel_poses[i, 0], rel_poses[i, 1], 'o', color=step_color, markersize=5)

    ax.plot(rel_poses[:, 0], rel_poses[:, 1], '-', color='blue', alpha=0.3, label="Drone path")

    lim = max_dist * 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend()

    # === Гистограмма ===
    ax = ax_hist
    ax.set_title("Step-wise rewards")
    ax.grid(True)

    bar_width = 0.2
    spacing = 0.05
    x_pos = []

    rewards = list(zip(
        saver.rewards_target_history,
        saver.rewards_obstacles_history,
        saver.step_penalty_history
    ))

    reward_colors = {
        'target': 'green',
        'obstacle': 'red',
        'step': 'gray'
    }

    for i, (rt, ro, sp) in enumerate(rewards):
        step_color = cmap_steps(i)
        base_x = i * (3 * (bar_width + spacing) + 0.2)
        x_pos.append(base_x)

        # Отрисовка столбцов без обводки
        ax.bar(base_x, rt / 10.0, width=bar_width, color=reward_colors['target'])
        ax.bar(base_x + bar_width + spacing, ro / 10.0, width=bar_width, color=reward_colors['obstacle'])
        ax.bar(base_x + 2 * (bar_width + spacing), sp / 10.0, width=bar_width, color=reward_colors['step'])

        # Шарик цвета шага под тремя столбцами
        center_x = base_x + (1.0 * (bar_width + spacing))  # центр группы
        ax.plot(center_x, -0.5, 'o', color=step_color, markersize=6)

    ax.set_xticks([])
    ax.set_ylabel("Reward components (scaled)")
    ax.set_xlabel("Steps (color-coded below)")

    legend_elements = [
        Patch(facecolor=reward_colors['target'], label='Reward for goal'),
        Patch(facecolor=reward_colors['obstacle'], label='Penalty for obstacles'),
        Patch(facecolor=reward_colors['step'], label='Step penalty')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    filename = f"{PATHS_DIR}/path{episode+1:04d}.png"
    plt.savefig(filename)
    plt.close()