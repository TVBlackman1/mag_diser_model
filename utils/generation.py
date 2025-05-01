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
    fixed_distance = field_size / 2.2

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
    obstacles = [
        tuple(np.random.uniform(0.0, field_size, size=2))
        for _ in range(num_obstacles)
    ]

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
