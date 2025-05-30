import numpy as np

from config.drone_config import DRONE_COLLISION_RADIUS
from config.env_config import TARGET_COLLISION_RADIUS, OBSTACLE_COLLISION_MAX_RADIUS, TARGET_COLLISION_MAX_RADIUS

obs_force_scale = 190.0
obs_range_scale = 0.64
obs_max_penalty = 1000

eps = 1e-6

def get_exponent_force_vector(a_pos, b_pos, collision_radius, range_scale, max_force, value_ratio, direction):
    dist = np.linalg.norm(a_pos - b_pos)
    norm = (a_pos - b_pos) / max(dist, eps)
    if dist < collision_radius:
        dist = collision_radius + eps
    value = (-collision_radius + dist) / range_scale
    force_value = value_ratio * np.exp(-value)
    force_value = min(force_value, max_force)

    if dist >= collision_radius * 3:
        force_value = 0
    if direction == 'in':
        norm = -norm
    return force_value, norm

def get_obs_force_field(drone_pos, obstacle_pos):
    return get_exponent_force_vector(
        drone_pos, obstacle_pos,
        OBSTACLE_COLLISION_MAX_RADIUS, obs_range_scale, obs_max_penalty,
        obs_force_scale,
        'out')

target_near_force_scale = 100.0
target_near_range_scale = 1.0
target_max_reward = 800


def get_target_force_near_field(drone_pos, target_pos):
    return get_exponent_force_vector(
        drone_pos, target_pos,
        TARGET_COLLISION_MAX_RADIUS, target_near_range_scale, target_max_reward,
        target_near_force_scale,
        'in')

target_standard_scale = 1.0

def get_target_force_standard_field(drone_pos, target_pos):
    dist = np.linalg.norm(drone_pos - target_pos)
    norm = (drone_pos - target_pos) / max(dist, eps)
    return target_standard_scale, -norm

def get_target_force_field(drone_pos, target_pos, near_threshold=1.5):
    dist = np.linalg.norm(drone_pos - target_pos)

    if dist > near_threshold:
        norm = (target_pos - drone_pos) / max(dist, eps)
        return 1.0, norm
    else:
        val_near, dir_near = get_target_force_near_field(drone_pos, target_pos)
        return val_near, dir_near

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
