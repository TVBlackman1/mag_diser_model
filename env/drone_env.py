import typing

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.barrier import get_obs_force_field, get_target_force_near_field
from utils.checks import is_target_reached, is_collision
from utils.generation import generate_environment_categorial
from config.drone_config import DRONE_SPEED
from config.env_config import FIELD_SIZE, NUM_OBSTACLES, STEP_PENALTY_MULTIPLIER


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.field_size = FIELD_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.drone_speed = DRONE_SPEED

        self.max_distance = np.sqrt(2 * self.field_size ** 2)

        self.step_number = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # [cos(angle), sin(angle), exist_flag, normalized_distance]
        obs_count = 5
        partical_obs_low = [-1.0, -1.0, 0.0, 0.0]
        partical_obs_high = [1.0, 1.0, 1.0, 1.0]
        self.observation_space = spaces.Box(
            low=np.array(partical_obs_low*obs_count, dtype=np.float32),
            high=np.array(partical_obs_high*obs_count, dtype=np.float32),
            shape=(len(partical_obs_low)*obs_count,),
            dtype=np.float32
        )

        self.reset()

    def set_positions(self, drone_pos, target_pos, obstacles):
        self.num_obstacles = len(obstacles)
        self.obstacles = obstacles
        self.drone_pos = drone_pos
        self.target_pos = target_pos
        self.last_distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)

    def get_current_obs(self):
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        level = options['level_difficult'] if (options is not None) else 'easy'
        env_data = generate_environment_categorial(self.field_size, level)

        self.drone_pos : np.ndarray= env_data["drone_pos"]
        self.target_pos : np.ndarray = env_data["target_pos"]
        self.obstacles : typing.Tuple[np.ndarray] = env_data["obstacles"]

        self.last_distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)
        self.step_number = 0

        return self._get_obs(), {}

    def step(self, move):
        self.step_number += 1
        target_reward = 0.0
        step_penalty = 0.0
        obstacle_penalty = 0.0
        result = 'process'

        move = move.detach().numpy()
        move = move / np.linalg.norm(move)

        new_pos = self.drone_pos + move * self.drone_speed

        new_pos = np.clip(new_pos, 0.0, self.field_size)

        hit_obstacle = is_collision(new_pos, self.obstacles)
        hit_target = is_target_reached(new_pos, self.target_pos)

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
            current_distance = np.linalg.norm(new_pos - self.target_pos)

            for obs in self.obstacles:
                value, dir = get_obs_force_field(new_pos, obs)
                dir = dir / np.linalg.norm(dir)
                alignment = np.dot(move, dir)
                if alignment > 0:
                    obstacle_penalty += alignment * value
            reward += obstacle_penalty

            if current_distance > 1e-5:
                value, _ = get_target_force_near_field(new_pos, self.target_pos)
                target_reward += value

                vector = self.target_pos - new_pos
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
        self.drone_pos = new_pos
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
        obstacles = sorted(self.obstacles, key=lambda _obs: np.linalg.norm(_obs - self.drone_pos))[:top]

        fill_array = [0 for i in range((top+1) * parameters)]
        self._get_partical_obs(self.target_pos, fill_array, 0)
        for i, obs in enumerate(obstacles):
            self._get_partical_obs(obs, fill_array, (i+1) * parameters)
        obs = np.array(fill_array, dtype=np.float32)
        return obs

    def _get_partical_obs(self, object_pos, fill_np_array, start_index=0, exist=True):
        direction_vector = object_pos - self.drone_pos
        distance = np.linalg.norm(direction_vector)

        exist_flag = 1.0 if exist else 0.0

        if not exist or distance > 1e-5:
            cos_theta = direction_vector[1] / distance
            sin_theta = direction_vector[0] / distance
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
        print(f"Drone position: {self.drone_pos}")
        print(f"Target position: {self.target_pos}")
        print(f"Obstacles: {self.obstacles}")
