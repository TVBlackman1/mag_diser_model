import numpy as np
from config.env_config import OBSTACLE_COLLISION_MAX_RADIUS, TARGET_COLLISION_MAX_RADIUS


def is_target_reached(drone_pos, target_pos):
    distance = np.linalg.norm(drone_pos - target_pos)
    return distance < TARGET_COLLISION_MAX_RADIUS

def is_target_reached_in_movement(drone_pos, drone_pos2, target_pos):
    return circle_segment_intersect(
            target_pos[0], target_pos[1], TARGET_COLLISION_MAX_RADIUS,
            drone_pos[0], drone_pos[1], drone_pos2[0], drone_pos2[1])

def is_collision(drone_pos, obstacles):
    for obs in obstacles:
        if np.linalg.norm(drone_pos - np.array(obs)) < OBSTACLE_COLLISION_MAX_RADIUS:
            return True
    return False

def is_collision_in_movement(drone_pos, drone_pos2, obstacles):
    for obs in obstacles:
        if circle_segment_intersect(
            obs[0], obs[1], OBSTACLE_COLLISION_MAX_RADIUS,
            drone_pos[0], drone_pos[1], drone_pos2[0], drone_pos2[1]):
            return True
    return False


def circle_segment_intersect(cx, cy, r, x1, y1, x2, y2):
    # Функция для вычисления квадрата расстояния
    def distance_sq(px, py, qx, qy):
        return (px - qx)**2 + (py - qy)**2
    
    # Проверяем, лежат ли концы отрезка внутри окружности
    d1_sq = distance_sq(cx, cy, x1, y1)
    d2_sq = distance_sq(cx, cy, x2, y2)
    
    if d1_sq <= r**2 or d2_sq <= r**2:
        return True  # Хотя бы одна точка внутри окружности
    
    # Вектор отрезка
    dx = x2 - x1
    dy = y2 - y1
    
    # Длина отрезка в квадрате
    segment_len_sq = dx**2 + dy**2
    
    # Если отрезок вырожден в точку
    if segment_len_sq == 0:
        return d1_sq <= r**2
    
    # Параметр проекции центра на прямую отрезка
    t = ((cx - x1) * dx + (cy - y1) * dy) / segment_len_sq
    
    # Ограничиваем параметр отрезком [0,1]
    t = max(0, min(1, t))
    
    # Находим ближайшую точку на отрезке
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    
    # Проверяем расстояние до ближайшей точки
    dist_sq = distance_sq(cx, cy, nearest_x, nearest_y)
    
    # Также проверяем расстояние до концов отрезка (на случай, если проекция вне отрезка)
    return dist_sq <= r**2 or d1_sq <= r**2 or d2_sq <= r**2
