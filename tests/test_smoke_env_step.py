import numpy as np

from config.env_config import FIELD_SIZE
from config.train_config import MAX_STEPS_PER_EPISODE
from env.drone_env import DroneEnv
from utils.generation import EnvGeneratorDifferentEpisodes


def test_env_can_reset_and_step():
    gen = EnvGeneratorDifferentEpisodes(FIELD_SIZE, max_episodes=10, max_steps=MAX_STEPS_PER_EPISODE)
    gen.set_state(episode=0, step=0, difficult_level="easy")

    env = DroneEnv(gen)
    obs, info = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)

    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)

    assert obs2.shape == env.observation_space.shape
    assert np.isfinite(float(reward))
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert isinstance(info2, dict)

