from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson, erlang, uniform
from agents.drone import Drone
from config.env_config import OBSTACLE_COLLISION_MAX_RADIUS
from typing import List
import numpy.typing as npt


@dataclass
class EnvData:
    """Container class for environment data in a simulation scenario.
    
    Attributes:
        drone: Current position of the drone as a NumPy array of float64 values.
                        May be None if position is not changed.
        target_position: Target position coordinates as a NumPy array of float64 values.
                         May be None if target is not changed.
        obstacles: List of obstacle positions, each represented by a NumPy array of float64 values.
                   May be None if positions and count are not changed.
    """
    drone: Drone | None
    target_position: npt.NDArray[np.float64] | None
    obstacles: List[npt.NDArray[np.float64]] | None

class EnvGenerator:
    def __init__(self, field_size, max_episodes, max_steps):
        self.field_size: float = field_size
        self.max_episodes: int = max_episodes
        self.max_steps: int = max_steps
        
        self.episode: int = -1
        self.step = -1
        self.level = ''
        
    def generate(self, current_env: EnvData) -> EnvData:
        raise "Not implemented environment generator"

    def set_state(self, episode: int, step: int, difficult_level: str = ''):
        """
        Method to set environment state parameters.

        Args:
            episode (int): Current episode number.
            step (int): Current step within the episode.
            difficult_level (str): Difficulty level for environment configuration.\
            Set -1 for using default values.
        """
        self.episode = episode
        self.level = difficult_level
        self.step = step


class EnvGeneratorDifferentEpisodes(EnvGenerator):
    def __init__(self, field_size, max_episodes, max_steps):
        super().__init__(field_size, max_episodes, max_steps)
        
        self.last_episode: int = -1
        
    def generate(self, current_env: EnvData) -> EnvData:
        if self.episode == self.last_episode:
            return EnvData(drone=None, target_position=None, obstacles=None)
        self.last_episode = self.episode
        
        level = self.level
        if len(level) == 0:
            level = get_level_difficult(self.episode, self.max_episodes)
        num_obstacles = generation_difficult_levels[level]['num_obstacles']
        
        fixed_distance = self.field_size / np.random.uniform(1.3, 3.1)

        drone_pos = np.random.uniform(0.0, self.field_size, size=2)

        for _ in range(100):
            theta = np.random.uniform(0.0, 2 * np.pi)
            offset = np.array([np.cos(theta), np.sin(theta)]) * fixed_distance
            target_pos = drone_pos + offset

            if np.all(target_pos >= 0.0) and np.all(target_pos <= self.field_size):
                break
        else:
            target_pos = np.clip(drone_pos + np.array([fixed_distance, 0]), 0.0, self.field_size)

        max_obstacles_at_centered_path = 4

        other_obstacles = num_obstacles - max_obstacles_at_centered_path
        if other_obstacles < 0:
            other_obstacles = 0
        obstacles_at_centered_path = num_obstacles - other_obstacles
        obstacles = [
            np.random.uniform(0.0, self.field_size, size=2)
            for _ in range(other_obstacles)
        ]

        for _ in range(obstacles_at_centered_path):
            alpha = np.random.uniform(0.3, 0.7)
            between_point = drone_pos + alpha * (target_pos - drone_pos)

            noise = np.random.normal(scale=0.03 * self.field_size, size=2)
            between_obstacle = np.clip(between_point + noise, 0.0, self.field_size)

            obstacles.append(between_obstacle)

        return EnvData(
            drone=Drone(pos=drone_pos),
            target_position=target_pos,
            obstacles=obstacles
        )


class EnvGeneratorDynamic(EnvGenerator):
    def __init__(self, field_size, max_episodes, max_steps):
        super().__init__(field_size, max_episodes, max_steps)
        
        mu = 1.3
        self.dist_obstacle_count = poisson(mu)
        self.dist_obstacle_angle = uniform(0, 2*np.pi)
        self.dist_obstacle_distance = erlang(3.0, 0.9)
        
        self.last_episode = -1
        
    def generate(self, current_env: EnvData) -> EnvData:
        change_all = self.last_episode != self.episode
        self.last_episode = self.episode
        
        if change_all:
            drone_pos = np.array([self.field_size/2, self.field_size/2])
            drone = Drone(drone_pos)
            target_pos = get_target_position(drone_pos, self.field_size)
        else:
            drone = None
            target_pos = None
        
        if drone is None:
            obstacles = self.get_obstacles(current_env.drone)
        else:
            obstacles = self.get_obstacles(drone)

        return EnvData(
            drone=drone,
            target_position=target_pos,
            obstacles=obstacles,
        )
    
    def get_obstacle_count(self):
        return int(self.dist_obstacle_count.rvs())
    
    def get_obstacles(self, drone: Drone) -> List[np.ndarray]:
        count = self.get_obstacle_count()
        ret = []
        
        start_point = drone.position
        min_shift_from_drone = OBSTACLE_COLLISION_MAX_RADIUS + 0.1
        for _ in range(count):
            angle = self.dist_obstacle_angle.rvs() + drone.orientation
            distance = self.dist_obstacle_distance.rvs() + min_shift_from_drone
            x_shift = distance * np.cos(angle)
            y_shift = distance * np.sin(angle)
            ret.append(np.array([
                start_point[0] + x_shift,
                start_point[1] + y_shift,
            ]))
        return ret
            

def get_target_position(drone_pos: npt.NDArray[np.float64], field_size: float) -> np.ndarray:
    fixed_distance = field_size / np.random.uniform(1.3, 3.1)
    for _ in range(100):
        theta = np.random.uniform(0.0, 2 * np.pi)
        offset = np.array([np.cos(theta), np.sin(theta)]) * fixed_distance
        target_pos = drone_pos + offset

        if np.all(target_pos >= 0.0) and np.all(target_pos <= field_size):
            break
    else:
        target_pos = np.clip(drone_pos + np.array([fixed_distance, 0]), 0.0, field_size)
        
    return target_pos


def get_level_difficult(episode: int, max_episode: int):
    level_difficult = 'easy'
    if episode / max_episode >= 0.4 or episode > 300:
        level_difficult = 'medium'
    return level_difficult

generation_difficult_levels = {
    'easy': {
        'num_obstacles': 0,
    },
    'medium': {
        'num_obstacles': 12,
    },
    'warmup': {
        'num_obstacles': 4,
    },
    'warmup-obs': {
        'num_obstacles': 30,
    }
}