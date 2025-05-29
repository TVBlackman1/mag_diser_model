import random

import numpy as np
from matplotlib import pyplot as plt

from config.env_config import FIELD_SIZE
from utils.generation import generate_environment_categorial

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

env = generate_environment_categorial(FIELD_SIZE, 'medium')
print(env)

# === Конфигурация поля ===
k_att = 1.0
k_rep = 100.0
d0 = 10.0
epsilon = 1e-6

OBSTACLE_COLLISION_RADIUS = 0.4
TARGET_COLLISION_RADIUS = 0.2
DRONE_COLLISION_RADIUS = 0.1

OBSTACLE_COLLISION_MAX_RADIUS=max(OBSTACLE_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)
TARGET_COLLISION_MAX_RADIUS=max(TARGET_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)


obs_force_scale = 150.0
obs_range_scale = 0.5

def get_obs_force_field(drone_pos, obstacle_pos):
    dist = np.linalg.norm(drone_pos - obstacle_pos)
    norm = (drone_pos - obstacle_pos) / dist
    value = (-OBSTACLE_COLLISION_MAX_RADIUS + dist)/obs_range_scale
    return obs_force_scale * np.exp(-value), norm

target_near_force_scale = 100.0
target_near_range_scale = 1.0


def get_target_force_near_field(drone_pos, target_pos):
    dist = np.linalg.norm(drone_pos - target_pos)
    norm = (drone_pos - target_pos) / dist
    value = (-TARGET_COLLISION_MAX_RADIUS + dist)/target_near_range_scale
    return target_near_force_scale * np.exp(-value), -norm

target_standard_scale = 1.0

def get_target_force_standard_field(drone_pos, target_pos):
    dist = np.linalg.norm(drone_pos - target_pos)
    norm = (drone_pos - target_pos) / dist
    return target_standard_scale, -norm

def get_target_force_field(drone_pos, target_pos):
    dir1, norm1 = get_target_force_near_field(drone_pos, target_pos)
    dir2, norm2 = get_target_force_standard_field(drone_pos, target_pos)
    force_vector = dir1 * norm1 + dir2 * norm2
    norm_force_vector = np.linalg.norm(force_vector)
    dir_force_vector = force_vector / norm_force_vector
    return dir_force_vector, norm_force_vector


# === Потенциальное поле ===
def compute_normalized_force(x, goal_pos, obstacles):
    total_force = np.zeros(2)

    value, dir = get_target_force_field(x, goal_pos)
    total_force += dir * value

    for obs in obstacles:
        value, dir =  get_obs_force_field(x, obs)
        total_force += dir * value

    norm = np.linalg.norm(total_force)
    if norm > 0:
        return total_force / norm, norm
    else:
        return np.zeros(2), 0

# === Построение графика ===
plt.figure(figsize=(10, 6))
plt.title('World barrier function (relative to drone)')
plt.grid(True)

# Дрон — в начале координат
plt.plot(0, 0, 'bo', label="Drone")

max_dist = 1.0
# Относительное положение цели
if env['target_pos'] is not None:
    rel_target = np.array(env['target_pos']) - env['drone_pos']
    plt.plot(rel_target[0], rel_target[1], 'ro', label="Target")
    max_dist = max(max_dist, float(np.linalg.norm(rel_target)))

# Относительные позиции препятствий
rel_obstacles = []
if env['obstacles']:
    for i, obs in enumerate(env['obstacles']):
        rel_obs = np.array(obs) - env['drone_pos']
        max_dist = max(max_dist, np.linalg.norm(rel_obs))
        rel_obstacles.append(rel_obs)
        plt.plot(rel_obs[0], rel_obs[1], 'kx', label="Obstacle" if i == 0 else None)

lim = max_dist * 1.2

# === Визуализация поля ===
portrait_resolution = 56
world_grid = np.linspace(-lim, lim, portrait_resolution)
X, Y = np.meshgrid(world_grid, world_grid)

U = np.zeros_like(X)
V = np.zeros_like(Y)
M = np.zeros_like(X)  # сила поля (модуль вектора)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        point = np.array([X[i, j], Y[i, j]])
        direction, magnitude = compute_normalized_force(point, rel_target, rel_obstacles)
        U[i, j] = direction[0]
        V[i, j] = direction[1]
        M[i, j] = magnitude

# Рисуем векторное поле с цветом по силе поля
quiv = plt.quiver(X, Y, U, V, M, cmap='cool', scale=90)

# Добавим цветовую шкалу справа
cbar = plt.colorbar(quiv)
cbar.set_label('Field Strength (|F|)', rotation=270, labelpad=15)
cbar.ax.tick_params(labelsize=8)

plt.xlim(-lim, lim)
plt.ylim(-lim, lim)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()