import numpy as np

from agents.drone import Drone
from config.env_config import FIELD_SIZE
from config.train_config import MAX_STEPS_PER_EPISODE
from env.drone_env import DroneEnv
from utils.generation import EnvData, EnvGenerator


class _FixedSceneGenerator(EnvGenerator):
    """Fixed scene generator: keeps target and obstacles constant; uses env RNG only for reproducibility wiring."""

    def __init__(self, field_size: float, start: np.ndarray, target: np.ndarray):
        super().__init__(field_size=field_size, max_episodes=10_000, max_steps=MAX_STEPS_PER_EPISODE)
        self._start = np.array(start, dtype=float)
        self._target = np.array(target, dtype=float)
        self.episode = 0

    def generate(self, current_env: EnvData) -> EnvData:
        # Always reset the drone to the same start pose for determinism.
        return EnvData(
            drone=Drone(pos=self._start.copy(), orientation=0.0),
            target_position=self._target.copy(),
            obstacles=[],
        )


def test_progress_reward_positive_when_moving_towards_target():
    # Place target ahead (+X). Positive speed should reduce distance.
    gen = _FixedSceneGenerator(FIELD_SIZE, start=np.array([0.0, 0.0]), target=np.array([100.0, 0.0]))
    env = DroneEnv(gen)
    obs, _ = env.reset(seed=1)

    # Move forward, no turning.
    obs2, reward_towards, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert not terminated
    assert not truncated
    assert info["new_distance"] < info["last_distance"]
    assert reward_towards > 0.0


def test_progress_reward_negative_when_moving_away_from_target():
    # Place target behind (-X). With orientation=0, forward movement increases distance.
    gen = _FixedSceneGenerator(FIELD_SIZE, start=np.array([0.0, 0.0]), target=np.array([-100.0, 0.0]))
    env = DroneEnv(gen)
    obs, _ = env.reset(seed=1)

    obs2, reward_away, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert not terminated
    assert not truncated
    assert info["new_distance"] > info["last_distance"]
    assert reward_away < 0.0


def test_episode_truncates_at_max_steps_with_timeout_result():
    # Target far away; take zero actions so we never reach it.
    gen = _FixedSceneGenerator(FIELD_SIZE, start=np.array([0.0, 0.0]), target=np.array([FIELD_SIZE, FIELD_SIZE]))
    env = DroneEnv(gen)
    env.reset(seed=1)

    last_info = None
    terminated = False
    truncated = False
    for _ in range(MAX_STEPS_PER_EPISODE):
        _, _, terminated, truncated, last_info = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if terminated or truncated:
            break

    assert not terminated
    assert truncated
    assert last_info is not None
    assert last_info["result"] == "timeout"

