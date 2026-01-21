from config.drone_config import DRONE_COLLISION_RADIUS
import math

FIELD_SIZE = 600.0    # meters
NUM_OBSTACLES = 4          # количество препятствий
TARGET_COLLISION_RADIUS = 0.3         # радиус достижения цели (м)
OBSTACLE_COLLISION_RADIUS = 0.4

DELTA_T = 0.1 # seconds

DISTANCE_REWARD_MULTIPLIER = 20.0  # коэффициент награды за приближение к цели
STEP_PENALTY_MULTIPLIER = 1   # или 0.3 - 0.7 для более мягких штрафов

OBSTACLE_COLLISION_MAX_RADIUS=max(OBSTACLE_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)
TARGET_COLLISION_MAX_RADIUS=max(TARGET_COLLISION_RADIUS, DRONE_COLLISION_RADIUS)

# Target movement parameters
TARGET_MAX_SPEED = 10.0  # m/s - maximum speed of moving target
TARGET_MAX_ANGULAR_VELOCITY = math.radians(45)  # radians/sec - maximum angular velocity
TARGET_MIN_DISTANCE_TO_OBSTACLE = OBSTACLE_COLLISION_MAX_RADIUS + 0.5  # meters - minimum distance from obstacles

# Obstacle curriculum (difficulty schedule)
# At these fractions of training progress, difficulty level increases.
# Used by generators that consult `get_level_difficult()`.
OBSTACLE_MEDIUM_START_FRAC = 0.4
OBSTACLE_HARD_START_FRAC = 0.8

# Dynamic generator: expected obstacle count near the drone (Poisson mean) per level
DYNAMIC_OBSTACLE_MU_EASY = 1.3
DYNAMIC_OBSTACLE_MU_MEDIUM = 4.0
DYNAMIC_OBSTACLE_MU_HARD = 8.0
