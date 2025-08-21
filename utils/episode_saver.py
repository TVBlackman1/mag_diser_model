from env.drone_env import DroneEnv

class EpisodeSaver:
    def __init__(self):
        pass

    def add_rewards(self, rewards_target, reward_obstacles, step_penalty):
        self.rewards_target_history.append(rewards_target)
        self.rewards_obstacles_history.append(reward_obstacles)
        self.step_penalty_history.append(step_penalty)

    def add_drone_pos(self, drone_pos):
        self.drone_poses.append(drone_pos)

    def start_episode(self, env: DroneEnv):
        self.rewards_target_history = []
        self.rewards_obstacles_history = []
        self.step_penalty_history = []

        self.drone_poses = [env.env_data.drone.position[:]]
        self.target_pos = env.env_data.target_position[:]
        self.obstacles = env.env_data.obstacles[:]