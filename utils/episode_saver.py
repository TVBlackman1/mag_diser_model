from env.drone_env import DroneEnv

class EpisodeSaverStatic:
    def __init__(self):
        self.episode = -1

    def add_rewards(self, rewards_target, reward_obstacles, step_penalty):
        self.rewards_target_history.append(rewards_target)
        self.rewards_obstacles_history.append(reward_obstacles)
        self.step_penalty_history.append(step_penalty)

    def add_drone_pos(self, drone_pos):
        self.drone_poses.append(drone_pos)

    def setup(self, env: DroneEnv, episode: int):
        if self.episode == episode:
            return
        self.episode = episode
        self.drone_poses = [env.env_data.drone.position[:]]
        self.target_pos = env.env_data.target_position[:]
        self.obstacles = env.env_data.obstacles[:]

        self.rewards_target_history = []
        self.rewards_obstacles_history = []
        self.step_penalty_history = []