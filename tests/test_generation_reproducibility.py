import numpy as np

from config.env_config import FIELD_SIZE
from config.train_config import MAX_STEPS_PER_EPISODE
from env.drone_env import DroneEnv
from utils.generation import EnvGeneratorDifferentEpisodes, EnvGeneratorDynamic


def test_different_episodes_generator_reproducible_with_seed():
    gen1 = EnvGeneratorDifferentEpisodes(FIELD_SIZE, max_episodes=10, max_steps=MAX_STEPS_PER_EPISODE)
    gen1.set_state(episode=0, step=0, difficult_level="easy")
    env1 = DroneEnv(gen1)
    env1.reset(seed=123)

    gen2 = EnvGeneratorDifferentEpisodes(FIELD_SIZE, max_episodes=10, max_steps=MAX_STEPS_PER_EPISODE)
    gen2.set_state(episode=0, step=0, difficult_level="easy")
    env2 = DroneEnv(gen2)
    env2.reset(seed=123)

    assert np.allclose(env1.env_data.drone.position, env2.env_data.drone.position)
    assert np.allclose(env1.env_data.target_position, env2.env_data.target_position)
    # easy => no obstacles
    assert env1.env_data.obstacles == env2.env_data.obstacles == []


def test_dynamic_generator_obstacles_clipped_to_bounds():
    gen = EnvGeneratorDynamic(FIELD_SIZE, max_episodes=10, max_steps=MAX_STEPS_PER_EPISODE)
    # Increase obstacle count likelihood for this test
    from scipy.stats import poisson

    gen.dist_obstacle_count = poisson(10.0)
    gen.set_state(episode=0, step=0, difficult_level="")

    env = DroneEnv(gen)
    env.reset(seed=123)

    # Force obstacle generation by setting step divisible by 6 and regenerating.
    gen.set_state(episode=0, step=6, difficult_level="")
    _ = gen.generate(env.env_data)
    if env.env_data.obstacles:
        obs = np.array(env.env_data.obstacles, dtype=float)
        assert (obs >= 0.0).all()
        assert (obs <= FIELD_SIZE).all()

