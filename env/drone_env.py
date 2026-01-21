from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium import spaces

from utils.barrier import get_obs_force_field, get_target_force_near_field
from utils.checks import is_collision_in_movement, is_target_reached_in_movement
from config.env_config import (
    FIELD_SIZE,
    NUM_OBSTACLES,
    STEP_PENALTY_MULTIPLIER,
    DELTA_T,
    DISTANCE_REWARD_MULTIPLIER,
    OBSTACLE_COLLISION_RADIUS,
    OBSTACLE_COLLISION_MAX_RADIUS,
)
from config.drone_config import DRONE_COLLISION_RADIUS, DRONE_MAX_FORWARD_SPEED
from config.train_config import MAX_STEPS_PER_EPISODE, NUM_EPISODES
from agents.drone import Drone
from utils.generation import EnvData, EnvGenerator
from utils.drone_view import AffineTransform2D
from utils.target_strategies import TargetStrategy


class DroneEnv(gym.Env):
    def __init__(self, generator: EnvGenerator, target_strategy: TargetStrategy | None = None):
        super(DroneEnv, self).__init__()

        self.field_size = FIELD_SIZE

        self.max_distance = np.sqrt(2 * self.field_size ** 2)

        self.step_number = 0
        self.delta_time = DELTA_T
        self.env_generator: EnvGenerator = generator
        self.target_strategy: TargetStrategy | None = target_strategy
        self.env_data: EnvData | None = None

        self.action_space = action_space()
        self.observation_space = observation_space()

        self.last_episode = -1
        self.last_distance_to_target = -1

    def get_current_obs(self):
        return self._get_obs()
    
    def reset(self, *, seed=None, options=None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # Ensure scenario generation is reproducible w.r.t. env seed.
        if hasattr(self.env_generator, "set_rng"):
            self.env_generator.set_rng(self.np_random)
        
        new_env_data = self.env_generator.generate(self.env_data)
        self._update_env_data(new_env_data)

        # Ensure target starts in a valid position for moving-target scenarios.
        if self.target_strategy is not None and self.env_data.target_position is not None:
            self.env_data.target_position = self.target_strategy.ensure_valid_position(
                self.env_data.target_position.copy(),
                self.env_data.obstacles or [],
                # Strategy instances already carry min_distance; reuse a conservative default via obstacles' radius.
                # Tests for target strategies pass an explicit min distance into ensure_valid_position directly.
                # Here we rely on the strategy implementation to keep it safe in update().
                getattr(self.target_strategy, "min_distance_to_obstacle", OBSTACLE_COLLISION_MAX_RADIUS + 0.5),
            )

        self.last_distance_to_target = np.linalg.norm(
            self.env_data.drone.position - self.env_data.target_position
        )
        # In Gymnasium/SB3 semantics, reset() starts a new episode.
        self.last_episode = self.env_generator.episode
        self.step_number = 0

        return self._get_obs(), {}

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_number += 1
        target_reward = 0.0
        step_penalty = 0.0
        obstacle_penalty = 0.0
        result = 'process'

        # Handle both torch tensors (old code) and numpy arrays (SB3)
        if hasattr(action, 'detach'):
            action = action.detach().cpu().numpy()
        elif isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        else:
            action = np.array(action, dtype=np.float32)
        
        speed_ratio, angle_ratio = float(action[0]), float(action[1])

        # Move target first (based on current env state) if strategy is enabled.
        # This avoids "catching" a target that would have moved away in the same step.
        if self.target_strategy is not None and self.env_data.target_position is not None:
            self.env_data.target_position = self.target_strategy.update(
                self.env_data.target_position.copy(),
                self.env_data,
                self.delta_time,
            )

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
        truncated = False

        last_distance = -1
        new_distance = -1

        if hit_obstacle:
            # Too-large terminal penalties make "do nothing" optimal.
            # Keep it strong, but comparable to possible positive return in an episode.
            reward = -300.0
            result = 'fail'
            terminated = True

        elif hit_target:
            reward = 300.0
            result = 'success'
            terminated = True

        else:
            old_distance_to_target = float(np.linalg.norm(old_position - self.env_data.target_position))
            distance_to_target = float(np.linalg.norm(position - self.env_data.target_position))

            # Per-step penalty: discourages stalling but should not dominate learning signal.
            step_penalty += float(STEP_PENALTY_MULTIPLIER) / float(MAX_STEPS_PER_EPISODE)

            progress = old_distance_to_target - distance_to_target
            normalized = progress / (DRONE_MAX_FORWARD_SPEED * self.delta_time)
            # Keep progress sign: moving away should be penalized.
            target_reward += float(DISTANCE_REWARD_MULTIPLIER) * float(np.clip(normalized, -1.0, 1.0))

            # if old_distance_to_target - distance_to_target < DRONE_MAX_FORWARD_SPEED * self.delta_time * 0.10:
            #     step_penalty += 1
            # else:
            #     target_reward += np.sqrt((old_distance_to_target - distance_to_target) / (DRONE_MAX_FORWARD_SPEED * self.delta_time))

            reward = target_reward - step_penalty - obstacle_penalty

            # Обновляем дистанцию до цели
            last_distance = old_distance_to_target
            new_distance = distance_to_target

        # Time limit: truncate episode if max steps reached (SB3 expects truncated flag).
        if not terminated and self.step_number >= MAX_STEPS_PER_EPISODE:
            truncated = True
            result = 'timeout'

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {
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
