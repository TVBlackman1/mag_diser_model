from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium import spaces

from utils.barrier import get_obs_force_field, get_target_force_near_field
from utils.checks import is_collision_in_movement, is_target_reached_in_movement
from config.env_config import (
    FIELD_SIZE, NUM_OBSTACLES, STEP_PENALTY_MULTIPLIER, DELTA_T, OBSTACLE_COLLISION_RADIUS, OBSTACLE_COLLISION_MAX_RADIUS
)
from config.drone_config import DRONE_COLLISION_RADIUS, DRONE_FORWARD_SPEED
from config.train_config import MAX_STEPS_PER_EPISODE, NUM_EPISODES
from agents.drone import Drone
from utils.generation import EnvData, EnvGenerator
from utils.drone_view import AffineTransform2D


class DroneEnv(gym.Env):
    def __init__(self, generator: EnvGenerator):
        super(DroneEnv, self).__init__()

        self.field_size = FIELD_SIZE

        self.max_distance = np.sqrt(2 * self.field_size ** 2)

        self.step_number = 0
        self.delta_time = DELTA_T
        self.env_generator: EnvGenerator = generator
        self.env_data: EnvData | None = None

        self.action_space = action_space()
        self.observation_space = observation_space()

        self.last_episode = -1
        self.last_distance_to_target = -1

    def get_current_obs(self):
        return self._get_obs()
    
    def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        
        new_env_data = self.env_generator.generate(self.env_data)
        self._update_env_data(new_env_data)

        self.last_distance_to_target = np.linalg.norm(
            self.env_data.drone.position - self.env_data.target_position
        )
        if self.last_episode != self.env_generator.episode:
            self.last_episode = self.env_generator.episode
            self.step_number = 0

        return self._get_obs(), {}

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_number += 1
        target_reward = 0.0
        step_penalty = 0.0
        obstacle_penalty = 0.0
        result = 'process'

        action = action.detach().numpy()
        speed_ratio, angle_ratio = action[0], action[1]

        old_position = self.env_data.drone.position.copy()
        old_orientation = self.env_data.drone.orientation
        self.env_data.drone.move(speed_ratio, angle_ratio, self.delta_time)
        self.env_data.drone.clip_in_boundary(self.field_size)
        
        position = self.env_data.drone.position.copy()
        orientation = self.env_data.drone.orientation

        move = position - old_position

        hit_obstacle = is_collision_in_movement(old_position, position, self.env_data.obstacles)
        hit_target = is_target_reached_in_movement(old_position, position, self.env_data.target_position)

        reward = 0.0
        terminated = False

        last_distance = 0.0
        new_distance = 0.0

        if hit_obstacle:
            reward = -10000
            result = 'fail'
            terminated = True

        elif hit_target:
            reward = 300
            result = 'success'
            terminated = True

        else:
            current_distance = np.linalg.norm(position - self.env_data.target_position)

            acc_radius = DRONE_COLLISION_RADIUS + OBSTACLE_COLLISION_RADIUS
            # for obs in self.obstacles:
            #     dy = obs[1] - old_position[1]
            #     dx = obs[0] - old_position[0]
            #     angle_to_obstable = np.atan2(dy, dx)
                
            #     dist = np.linalg.norm(obs - old_position)
                # MAX_DISTANCE = DRONE_FORWARD_SPEED * self.delta_time
            #     if dist + OBSTACLE_COLLISION_MAX_RADIUS > 2:
            #         continue
                
            #     min_angle_from_obs_dir = np.arcsin(acc_radius/dist)
            #     if (
            #         orientation <= angle_to_obstable + min_angle_from_obs_dir or
            #         orientation >= angle_to_obstable - min_angle_from_obs_dir
            #     ):
            #         # obstacle_penalty = MAX_DISTANCE / (dist + OBSTACLE_COLLISION_MAX_RADIUS)
            #         obstacle_penalty = 2
                # value, dir = get_obs_force_field(position, obs)
                # dir = dir / np.linalg.norm(dir)
                # alignment = np.dot(move, dir)
                # if alignment > 0:
                #     obstacle_penalty += alignment * value
            # reward += obstacle_penalty

            # if current_distance > 1e-5:
            #     # value, _ = get_target_force_near_field(position, self.target_pos)
            #     # target_reward += value

            #     # vector = self.target_pos - position
            #     # dist = np.linalg.norm(vector)
            #     # vector = vector / dist
            #     # alignment = np.dot(move, vector)
            #     # if alignment > 0:
            #     #     target_reward += alignment * 2

            #     old_distanse = np.linalg.norm(old_position - self.target_pos)
            #     distanse = np.linalg.norm(position - self.target_pos)
            #     target_reward += np.pow((old_distanse - distanse), 1)
            #     if np.linalg.norm(move) < 0.7:
            #         target_reward *= -1

            old_distance = float(np.linalg.norm(old_position - self.env_data.target_position))
            distance = float(np.linalg.norm(position - self.env_data.target_position))

            # if np.linalg.norm(move) < 15 * self.delta_time * 0.2:
            #     step_penalty += 1
            step_penalty += 1 / MAX_STEPS_PER_EPISODE

            target_reward = (old_distance - distance) / old_distance
            if old_distance - distance < 15 * self.delta_time * 0.10:
                step_penalty += 1
            else:
                target_reward += np.sqrt((old_distance - distance) / (15 * self.delta_time))
            # target_reward = np.exp(-alpha * distance_to_target) - 1.0

            reward = target_reward - step_penalty - obstacle_penalty

            # reward += target_reward
            # step_penalty = -STEP_PENALTY_MULTIPLIER * 1
            # step_penalty = -STEP_PENALTY_MULTIPLIER * (current_distance / self.max_distance)
            # reward += step_penalty * (self.step_number * 12)

            # Обновляем дистанцию до цели
            last_distance = self.last_distance_to_target
            new_distance = current_distance
            self.last_distance_to_target = current_distance

        obs = self._get_obs()
        return obs, reward, terminated, False, {
            'target_reward': target_reward,
            'step_penalty': step_penalty,
            'obstacle_penalty': obstacle_penalty,
            'result': result,
            'last_distance': last_distance,
            'new_distance': new_distance,
        }

    def _get_obs(self):
        top = 4
        parameters = 4
        obstacles = sorted(self.env_data.obstacles,
            key=lambda _obs: np.linalg.norm(_obs - self.env_data.drone.position)
        )[:top]

        fill_array = [0 for i in range((top+1) * parameters)]

        transform = AffineTransform2D(self.env_data.drone.position, self.env_data.drone.orientation)
        self._get_partical_obs(transform, self.env_data.target_position, fill_array, 0)
        for i, obstacle in enumerate(obstacles):
            self._get_partical_obs(transform, obstacle, fill_array, (i+1) * parameters)
        obstacle = np.array(fill_array, dtype=np.float32)
        # TODO check is exist work without obstacles
        return obstacle

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

    def _update_env_data(self, new_env_data: EnvData):
        if self.env_data is None:
            self.env_data = new_env_data
        else:
            if new_env_data.drone is not None:
                self.env_data.drone = new_env_data.drone
            if new_env_data.target_position is not None:
                self.env_data.target_position = new_env_data.target_position
            if new_env_data.obstacles is not None:
                self.env_data.obstacles = new_env_data.obstacles
                
    def render(self):
        print(f"Drone position: {self.env_data.drone.position}")
        print(f"Target position: {self.env_data.target_position}")
        print(f"Obstacles: {self.env_data.obstacles}")

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
