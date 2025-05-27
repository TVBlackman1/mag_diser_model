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

# === Потенциальное поле ===
def compute_normalized_force(x, goal_pos, obstacles):
    dir_to_goal = x - goal_pos
    force_att = -k_att * dir_to_goal

    force_rep = np.zeros(2)
    for obs in obstacles:
        dir_from_obs = x - obs
        dist = np.linalg.norm(dir_from_obs)
        if dist < d0 and dist > epsilon:
            rep_strength = k_rep * (1.0 / dist - 1.0 / d0) / (dist**3 + epsilon)
            force_rep += rep_strength * dir_from_obs

    total_force = force_att + force_rep
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
quiv = plt.quiver(X, Y, U, V, M, cmap='coolwarm', scale=90)

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