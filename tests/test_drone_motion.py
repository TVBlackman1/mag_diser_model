import numpy as np

from agents.drone import Drone
from config.drone_config import DRONE_MAX_ANGULAR_VELOCITY, DRONE_MAX_FORWARD_SPEED


def test_drone_moves_forward_when_orientation_zero():
    drone = Drone(pos=np.array([0.0, 0.0], dtype=float), orientation=0.0)
    dt = 0.1

    drone.move(speed_ratio=1.0, angle_ratio=0.0, delta_time=dt)

    expected_dx = DRONE_MAX_FORWARD_SPEED * dt
    assert np.isclose(drone.position[0], expected_dx, atol=1e-6)
    assert np.isclose(drone.position[1], 0.0, atol=1e-6)
    assert np.isclose(drone.orientation, 0.0, atol=1e-6)


def test_drone_turn_changes_orientation_and_direction():
    drone = Drone(pos=np.array([0.0, 0.0], dtype=float), orientation=0.0)
    dt = 0.1

    drone.move(speed_ratio=1.0, angle_ratio=1.0, delta_time=dt)

    expected_theta = DRONE_MAX_ANGULAR_VELOCITY * dt
    expected_step = DRONE_MAX_FORWARD_SPEED * dt
    assert np.isclose(drone.orientation, expected_theta, atol=1e-6)
    assert np.isclose(drone.position[0], np.cos(expected_theta) * expected_step, atol=1e-6)
    assert np.isclose(drone.position[1], np.sin(expected_theta) * expected_step, atol=1e-6)

