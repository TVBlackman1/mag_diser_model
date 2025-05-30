import numpy as np
from config.env_config import OBSTACLE_COLLISION_MAX_RADIUS, TARGET_COLLISION_MAX_RADIUS


def is_target_reached(drone_pos, target_pos):
    distance = np.linalg.norm(drone_pos - target_pos)
    return distance < TARGET_COLLISION_MAX_RADIUS

def is_collision(drone_pos, obstacles):
    """
    Проверяет, произошло ли столкновение дрона с любым препятствием.
    """
    for obs in obstacles:
        if np.linalg.norm(drone_pos - np.array(obs)) < OBSTACLE_COLLISION_MAX_RADIUS:
            return True
    return False