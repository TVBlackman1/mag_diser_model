import numpy as np
from typing import List, Tuple, Dict

retp = False
def generate_environment(field_size: float, num_obstacles: int) -> Dict[str, object]:
    """Генерирует начальную конфигурацию среды: дрон и цель на фиксированном расстоянии (1/3 поля), препятствия."""
    # print("generating environment")
    global retp

    # if retp != False:
    #     return retp

    # print("??")
    fixed_distance = field_size / np.random.uniform(1.3, 3.1)

    # Генерируем позицию дрона
    drone_pos = np.random.uniform(0.0, field_size, size=2)

    # Генерируем позицию цели на фиксированном расстоянии
    for _ in range(100):  # максимум 100 попыток
        theta = np.random.uniform(0.0, 2 * np.pi)
        offset = np.array([np.cos(theta), np.sin(theta)]) * fixed_distance
        target_pos = drone_pos + offset

        if np.all(target_pos >= 0.0) and np.all(target_pos <= field_size):
            break
    else:
        # Если не нашли валидную позицию — зажмём в пределах
        target_pos = np.clip(drone_pos + np.array([fixed_distance, 0]), 0.0, field_size)

    # Генерируем препятствия
    # Генерируем препятствия
    obstacles = [
        tuple(np.random.uniform(0.0, field_size, size=2))
        for _ in range(num_obstacles - 1)
    ]

    # Добавляем хотя бы одно препятствие между дроном и целью
    if num_obstacles > 0:
        alpha = np.random.uniform(0.3, 0.7)  # не строго по центру
        between_point = drone_pos + alpha * (target_pos - drone_pos)

        # Добавим небольшой шум, чтобы не быть строго на линии
        noise = np.random.normal(scale=0.05 * field_size, size=2)
        between_obstacle = tuple(np.clip(between_point + noise, 0.0, field_size))

        obstacles.append(between_obstacle)

    retp = {
        "drone_pos": drone_pos,
        "target_pos": target_pos,
        "obstacles": obstacles
    }
    return {
        "drone_pos": drone_pos,
        "target_pos": target_pos,
        "obstacles": obstacles
    }

generation_difficult_levels = {
    'easy': {
        'num_obstacles': 0,
    },
    'medium': {
        'num_obstacles': 3,
    },
    'hard': {
        'num_obstacles': 10,
    }
}

def generate_environment_categorial(field_size: float, level: str) -> Dict[str, object]:
    return generate_environment(field_size, generation_difficult_levels[level]['num_obstacles'])