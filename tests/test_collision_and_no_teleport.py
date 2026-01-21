import numpy as np

from agents.drone import Drone
from config.drone_config import DRONE_MAX_FORWARD_SPEED
from config.env_config import DELTA_T, FIELD_SIZE, OBSTACLE_COLLISION_MAX_RADIUS, TARGET_COLLISION_MAX_RADIUS
from env.drone_env import DroneEnv
from utils.generation import EnvData, EnvGenerator


class _FixedScene(EnvGenerator):
    def __init__(self, env_data: EnvData):
        super().__init__(field_size=FIELD_SIZE, max_episodes=1, max_steps=10_000)
        self._env_data = env_data

    def generate(self, current_env: EnvData) -> EnvData:
        return self._env_data


def test_no_teleport_step_distance_limited_by_speed_and_dt():
    start = np.array([10.0, 10.0], dtype=float)
    drone = Drone(pos=start.copy(), orientation=0.0)
    target = np.array([1000.0, 10.0], dtype=float)  # far away, no terminal
    env = DroneEnv(_FixedScene(EnvData(drone=drone, target_position=target, obstacles=[])))
    env.reset(seed=1)

    old = env.env_data.drone.position.copy()
    env.step(np.array([1.0, 0.0], dtype=np.float32))
    new = env.env_data.drone.position.copy()

    step_dist = float(np.linalg.norm(new - old))
    max_step = float(DRONE_MAX_FORWARD_SPEED * DELTA_T) + 1e-9
    assert step_dist <= max_step


def test_collision_terminates_when_segment_hits_obstacle():
    # One forward step moves about v*dt = 1.5m.
    # Place obstacle on the segment within collision radius.
    start = np.array([10.0, 10.0], dtype=float)
    drone = Drone(pos=start.copy(), orientation=0.0)
    target = np.array([1000.0, 10.0], dtype=float)

    obstacle_center = start + np.array([0.75, 0.0], dtype=float)
    env = DroneEnv(_FixedScene(EnvData(drone=drone, target_position=target, obstacles=[obstacle_center])))
    env.reset(seed=1)

    _obs, _reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["result"] == "fail"


def test_target_reached_terminates_when_segment_hits_target():
    start = np.array([10.0, 10.0], dtype=float)
    drone = Drone(pos=start.copy(), orientation=0.0)

    # Put target center on the segment; no obstacles.
    target = start + np.array([0.75, 0.0], dtype=float)
    env = DroneEnv(_FixedScene(EnvData(drone=drone, target_position=target, obstacles=[])))
    env.reset(seed=1)

    _obs, _reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["result"] == "success"


def test_target_radius_is_independent_from_obstacle_radius():
    # Regression test for bug: is_target_reached_in_movement must use TARGET_COLLISION_MAX_RADIUS.
    # Construct a case where the segment passes within obstacle-radius but outside target-radius.
    # If code wrongly uses obstacle radius, it would incorrectly mark success.
    assert OBSTACLE_COLLISION_MAX_RADIUS >= TARGET_COLLISION_MAX_RADIUS

    start = np.array([10.0, 10.0], dtype=float)
    drone = Drone(pos=start.copy(), orientation=0.0)

    # Drone will move along +X. Place target slightly off the path:
    # distance from segment equals (TARGET_RADIUS + small_eps) but still < OBSTACLE_RADIUS.
    eps = 1e-3
    y_offset = float(TARGET_COLLISION_MAX_RADIUS + eps)
    target = start + np.array([0.75, y_offset], dtype=float)

    env = DroneEnv(_FixedScene(EnvData(drone=drone, target_position=target, obstacles=[])))
    env.reset(seed=1)

    _obs, _reward, terminated, truncated, info = env.step(np.array([1.0, 0.0], dtype=np.float32))
    assert not terminated, "Should not be success: target is outside target radius"
    assert info["result"] in ("process", "timeout")

