# env/drone_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils.generation import generate_environment
from config.drone_config import DRONE_SPEED, DRONE_COLLISION_RADIUS
from config.env_config import FIELD_SIZE, NUM_OBSTACLES, TARGET_RADIUS, DISTANCE_REWARD_MULTIPLIER, STEP_PENALTY_MULTIPLIER


class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        self.field_size = FIELD_SIZE
        self.num_obstacles = NUM_OBSTACLES
        self.drone_speed = DRONE_SPEED

        self.max_distance = np.sqrt(2 * self.field_size ** 2)  # диагональ поля, макс. дистанция

        # Действия: 8 направлений
        self.action_space = spaces.Discrete(8)

        # Наблюдения: [cos(angle), sin(angle), exist_flag, normalized_distance]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        env_data = generate_environment(self.field_size, self.num_obstacles)

        self.drone_pos = env_data["drone_pos"]
        self.target_pos = env_data["target_pos"]
        self.obstacles = env_data["obstacles"]

        self.last_distance_to_target = np.linalg.norm(self.drone_pos - self.target_pos)

        return self._get_obs(), {}

    def step(self, action):
        directions = [
            np.array([0, 1]), [0, -1], [-1, 0], [1, 0],
            [-1, 1], [1, 1], [-1, -1], [1, -1]
        ]
        move = directions[action]
        move = move / np.linalg.norm(move)

        new_pos = self.drone_pos + move * self.drone_speed
        new_pos = np.clip(new_pos, 0.0, self.field_size)

        hit_obstacle = any(np.linalg.norm(new_pos - np.array(obs)) < DRONE_COLLISION_RADIUS for obs in self.obstacles)
        hit_target = np.linalg.norm(new_pos - self.target_pos) < TARGET_RADIUS

        reward = 0.0
        terminated = False

        if hit_obstacle:
            reward = -10
            terminated = True
        elif hit_target:
            reward = 100
            terminated = True
        else:
            # Обычный ход — считаем динамическую награду
            current_distance = np.linalg.norm(new_pos - self.target_pos)
            delta_distance = self.last_distance_to_target - current_distance

            # Бонус за приближение (относительно прошлой дистанции)
            if self.last_distance_to_target > 1e-5:
                reward += (delta_distance / self.last_distance_to_target) * DISTANCE_REWARD_MULTIPLIER

            # Штраф за шаг пропорциональный расстоянию
            step_penalty = - STEP_PENALTY_MULTIPLIER * (current_distance / self.max_distance)
            reward += step_penalty

            # Обновляем дистанцию для следующего шага
            self.last_distance_to_target = current_distance

        self.drone_pos = new_pos

        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        direction_vector = self.target_pos - self.drone_pos
        distance = np.linalg.norm(direction_vector)

        if distance > 1e-5:  # чтобы избежать деления на ноль
            cos_theta = direction_vector[0] / distance
            sin_theta = direction_vector[1] / distance
            exist_flag = 1.0
            distance_normalized = np.clip(distance / self.max_distance, 0.0, 1.0)
        else:
            cos_theta = 0.0
            sin_theta = 0.0
            exist_flag = 0.0
            distance_normalized = 0.0

        obs = np.array([cos_theta, sin_theta, exist_flag, distance_normalized], dtype=np.float32)
        return obs

    def render(self):
        print(f"Drone position: {self.drone_pos}")
        print(f"Target position: {self.target_pos}")
        print(f"Obstacles: {self.obstacles}")
