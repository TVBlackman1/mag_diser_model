# Параметры обучения
NUM_EPISODES = 600
MAX_STEPS_PER_EPISODE = 300
EVAL_INTERVAL = 100  # каждые 50 эпизодов сохраняем модель

TRAIN_COUNT = 100
TEST_COUNT = 5

# Гиперпараметры агента
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005

# For off-policy (TD3/DDPG), too-small replay buffer makes learning unstable.
BUFFER_SIZE = int(200_000)
BATCH_SIZE = 256

# Noise (размах случайности для exploration)
# 0.5 is usually too aggressive here and pushes speed/angle into random thrashing.
ACTION_NOISE_STD = 0.2
ACTION_NOISE_STD2 = 0.1
ACTION_NOISE_STD3 = 0.03

# REWARD SCALE
OBS_FORCE_SCALE = 150.0
