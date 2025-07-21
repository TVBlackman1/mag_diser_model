import numpy as np


class EnvGenerator:
    def __init__(self, field_size, max_episodes, max_steps):
        self.field_size = field_size
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        
    def generate(self, episode=0, step=0, level="") -> dict[str, any]:
        raise "Not implemented environment generator"
    
    def change_probabilities():
        raise "Not implemented environment generator"


class EnvGeneratorDifferentEpisodes(EnvGenerator):
    def __init__(self, field_size, max_episodes, max_steps):
        super().__init__(field_size, max_episodes, max_steps)
        
        self.last_episode = -1
        self.last_positions = None
        
    def generate(self, episode=0, step=0, level="") -> dict[str, any]:
        if episode == self.last_episode:
            return self.last_positions
        
        if len(level) == 0:
            level = get_level_difficult(episode, self.max_episodes)
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
            tuple(np.random.uniform(0.0, self.field_size, size=2))
            for _ in range(other_obstacles)
        ]

        for _ in range(obstacles_at_centered_path):
            alpha = np.random.uniform(0.3, 0.7)
            between_point = drone_pos + alpha * (target_pos - drone_pos)

            noise = np.random.normal(scale=0.03 * self.field_size, size=2)
            between_obstacle = tuple(np.clip(between_point + noise, 0.0, self.field_size))

            obstacles.append(between_obstacle)

        episode_positions = {
            "drone_pos": drone_pos,
            "target_pos": target_pos,
            "obstacles": []
        }
        self.last_episode = episode
        self.last_positions = episode_positions
        return episode_positions


def get_level_difficult(episode, max_episode):
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