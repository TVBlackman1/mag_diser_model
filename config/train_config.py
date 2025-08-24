# Параметры обучения
NUM_EPISODES = 800
MAX_STEPS_PER_EPISODE = 200
EVAL_INTERVAL = 50  # каждые 50 эпизодов сохраняем модель

TRAIN_COUNT = 40
TEST_COUNT = 1

# Гиперпараметры агента
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.005

BUFFER_SIZE = int(16000)
BATCH_SIZE = 256

# Noise (размах случайности для exploration)
ACTION_NOISE_STD = 0.5
ACTION_NOISE_STD2 = 0.1
ACTION_NOISE_STD3 = 0.03

# REWARD SCALE
OBS_FORCE_SCALE = 150.0
