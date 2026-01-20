import numpy as np

from agents.drone import Drone
from config.env_config import FIELD_SIZE
from env.drone_env import DroneEnv
from utils.generation import EnvData, EnvGenerator


class _FixedGenerator(EnvGenerator):
    """Deterministic generator: always returns the same scene."""

    def __init__(self, field_size: float, env_data: EnvData):
        super().__init__(field_size=field_size, max_episodes=1, max_steps=1)
        self._fixed = env_data

    def generate(self, current_env: EnvData) -> EnvData:
        return self._fixed


def _make_env(drone: Drone, target_pos: np.ndarray, obstacles: list[np.ndarray]) -> DroneEnv:
    gen = _FixedGenerator(
        FIELD_SIZE,
        EnvData(drone=drone, target_position=np.array(target_pos, dtype=float), obstacles=[np.array(o, dtype=float) for o in obstacles]),
    )
    return DroneEnv(gen)


def test_observation_target_straight_ahead_orientation_zero():
    # With orientation=0, "ahead" in this coordinate convention is +X in world coordinates.
    drone = Drone(pos=np.array([0.0, 0.0], dtype=float), orientation=0.0)
    env = _make_env(drone=drone, target_pos=np.array([10.0, 0.0]), obstacles=[])

    obs, _ = env.reset(seed=123)
    # Target occupies first 4 entries.
    cos_theta, sin_theta, exist_flag, dist_norm = obs[0], obs[1], obs[2], obs[3]

    assert exist_flag == 1.0
    assert np.isclose(cos_theta, 1.0, atol=1e-6)
    assert np.isclose(sin_theta, 0.0, atol=1e-6)

    expected = 10.0 / env.max_distance
    assert np.isclose(dist_norm, expected, atol=1e-6)


def test_observation_target_left_gives_negative_sin():
    # For orientation=0, +Y world becomes "left", which should give cos=0, sin=-1.
    drone = Drone(pos=np.array([0.0, 0.0], dtype=float), orientation=0.0)
    env = _make_env(drone=drone, target_pos=np.array([0.0, 10.0]), obstacles=[])

    obs, _ = env.reset(seed=123)
    cos_theta, sin_theta = obs[0], obs[1]
    assert np.isclose(cos_theta, 0.0, atol=1e-6)
    assert np.isclose(sin_theta, -1.0, atol=1e-6)


def test_observation_obstacles_sorted_by_distance():
    drone = Drone(pos=np.array([0.0, 0.0], dtype=float), orientation=0.0)
    target = np.array([10.0, 0.0])
    # Distances: 1, 2, 3, 4 (in world units)
    obstacles = [
        np.array([4.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([3.0, 0.0]),
        np.array([2.0, 0.0]),
    ]
    env = _make_env(drone=drone, target_pos=target, obstacles=obstacles)

    obs, _ = env.reset(seed=123)

    def slot(i: int) -> np.ndarray:
        # slot 0 = target, slot 1..4 = obstacles, each 4 values
        start = i * 4
        return obs[start : start + 4]

    # Nearest obstacle is at distance 1.0 -> first obstacle slot is slot(1)
    o1 = slot(1)
    assert np.isclose(o1[2], 1.0, atol=1e-6)  # exist_flag
    expected_dist_norm = 1.0 / env.max_distance
    assert np.isclose(o1[3], expected_dist_norm, atol=1e-6)

