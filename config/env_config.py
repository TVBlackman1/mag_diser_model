from config.drone_config import DRONE_COLLISION_RADIUS

FIELD_SIZE = 10.0           # размер поля (м)
NUM_OBSTACLES = 4          # количество препятствий
TARGET_COLLISION_RADIUS = 0.3         # радиус достижения цели (м)
OBSTACLE_COLLISION_RADIUS = 0.4

DISTANCE_REWARD_MULTIPLIER = 20.0  # коэффициент награды за приближение к цели
STEP_PENALTY_MULTIPLIER = 1   # или 0.3 - 0.7 для более мягких штрафов

OBSTACLE_COLLISION_MAX_RADIUS=max(OBSTACLE_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)
TARGET_COLLISION_MAX_RADIUS=max(TARGET_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)
