import typing

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.barrier import get_obs_force_field, get_target_force_near_field
from utils.checks import is_target_reached, is_collision
from config.env_config import FIELD_SIZE, NUM_OBSTACLES, STEP_PENALTY_MULTIPLIER, DELTA_T
from agents.drone import Drone
from utils.generation import EnvGenerator
from utils.drone_view import AffineTransform2D


class DroneEnv(gym.Env):
    def __init__(self, generator: EnvGenerator):
        super(DroneEnv, self).__init__()

        self.field_size = FIELD_SIZE

        self.max_distance = np.sqrt(2 * self.field_size ** 2)

        self.step_number = 0
        self.delta_time = DELTA_T
        self.env_generator: EnvGenerator = generator

        self.action_space = action_space()
        self.observation_space = observation_space()

        self.reset()

    def set_positions(self, drone_pos, target_pos, obstacles):
        self.num_obstacles = len(obstacles)
        self.obstacles = obstacles
        self.drone: Drone = Drone(drone_pos)
        self.target_pos = target_pos
        self.last_distance_to_target = np.linalg.norm(self.drone.position - self.target_pos)

    def get_current_obs(self):
        return self._get_obs()

    def reset(self, episode=0, step=0, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.env_generator is None:
            self.target_pos = np.array([0., 0.])
            self.drone = Drone([0., 0.])
            self.obstacles = []
            return self._get_obs(), {}

        
        level = options['level_difficult'] if (options is not None) else ''
        env_data = self.env_generator.generate(episode, step, level)
        
        self.drone = Drone(env_data["drone_pos"])
        self.target_pos : np.ndarray = env_data["target_pos"]
        self.obstacles : typing.Tuple[np.ndarray] = env_data["obstacles"]

        self.last_distance_to_target = np.linalg.norm(self.drone.position - self.target_pos)
        self.step_number = 0

        return self._get_obs(), {}

    def step(self, action):
        self.step_number += 1
        target_reward = 0.0
        step_penalty = 0.0
        obstacle_penalty = 0.0
        result = 'process'

        action = action.detach().numpy()
        speed_ratio, angle_ratio = action[0], action[1]

        old_position = self.drone.position
        self.drone.move(speed_ratio, angle_ratio, self.delta_time)
        self.drone.clip_in_boundary(self.field_size)
        position = self.drone.position
        
        move = position - old_position
        
        hit_obstacle = is_collision(position, self.obstacles)
        hit_target = is_target_reached(position, self.target_pos)

        reward = 0.0
        terminated = False

        if hit_obstacle:
            reward = -10000
            result = 'fail'
            terminated = True

        elif hit_target:
            reward = 300
            result = 'success'
            terminated = True

        else:
            current_distance = np.linalg.norm(position - self.target_pos)

            for obs in self.obstacles:
                value, dir = get_obs_force_field(position, obs)
                dir = dir / np.linalg.norm(dir)
                alignment = np.dot(move, dir)
                if alignment > 0:
                    obstacle_penalty += alignment * value
            reward += obstacle_penalty

            if current_distance > 1e-5:
                value, _ = get_target_force_near_field(position, self.target_pos)
                target_reward += value

                vector = self.target_pos - position
                dist = np.linalg.norm(vector)
                vector = vector / dist
                alignment = np.dot(move, vector)
                if alignment > 0:
                    target_reward += alignment * 2
            reward += target_reward
            step_penalty = -STEP_PENALTY_MULTIPLIER * (current_distance / self.max_distance)
            reward += step_penalty * (self.step_number * 80)

            # Обновляем дистанцию до цели
            self.last_distance_to_target = current_distance

        obs = self._get_obs()
        return obs, reward, terminated, False, {
            'target_reward': target_reward,
            'step_penalty': step_penalty,
            'obstacle_penalty': obstacle_penalty,
            'result': result,
        }

    def _get_obs(self):
        top = 4
        parameters = 4
        obstacles = sorted(self.obstacles, key=lambda _obs: np.linalg.norm(_obs - self.drone.position))[:top]

        fill_array = [0 for i in range((top+1) * parameters)]
        
        transform = AffineTransform2D(self.drone.position, self.drone.orientation)
        self._get_partical_obs(transform, self.target_pos, fill_array, 0)
        for i, obs in enumerate(obstacles):
            self._get_partical_obs(transform, obs, fill_array, (i+1) * parameters)
        obs = np.array(fill_array, dtype=np.float32)
        return obs

    def _get_partical_obs(self, transform: AffineTransform2D, object_pos, fill_np_array, start_index=0, exist=True):
        relative_vector = transform.apply([object_pos])[:, 0]
        distance = np.linalg.norm(relative_vector)

        exist_flag = 1.0 if exist else 0.0
        if not exist or distance > 1e-5:
            cos_theta = relative_vector[1] / distance
            sin_theta = relative_vector[0] / distance
            distance_normalized = np.clip(distance / self.max_distance, 0.0, 1.0)
        else:
            cos_theta = 0.0
            sin_theta = 0.0
            distance_normalized = 0.0

        fill_np_array[start_index] = cos_theta
        fill_np_array[start_index+1] = sin_theta
        fill_np_array[start_index+2] = exist_flag
        fill_np_array[start_index+3] = distance_normalized

    def render(self):
        print(f"Drone position: {self.drone.position}")
        print(f"Target position: {self.target_pos}")
        print(f"Obstacles: {self.obstacles}")

def action_space():
    # [dx, dy] for drone
    return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

def observation_space():
    obstacle_count = 4
    objects_count = obstacle_count + 1 # with target observation
    
    # [cos(angle), sin(angle), exist_flag, normalized_distance]
    partical_obs_low = [-1.0, -1.0, 0.0, 0.0]
    partical_obs_high = [1.0, 1.0, 1.0, 1.0]
    
    observation_space = spaces.Box(
        low=np.array(partical_obs_low*objects_count, dtype=np.float32),
        high=np.array(partical_obs_high*objects_count, dtype=np.float32),
        shape=(len(partical_obs_low)*objects_count,),
        dtype=np.float32
    )
    
    return observation_space