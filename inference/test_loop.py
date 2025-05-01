import math
import torch
from agents.ddpg_agent import DDPGAgent, map_action_to_direction
from env.drone_env import DroneEnv
from inference import get_last
from inference.load_model import load_model

DEVICE = "cpu"

NUM_EPISODES = 5000
MAX_STEPS = 300

def main():
    model_path = get_last.get_latest_model_path()
    env, agent = load_model()

    successes = 0
    stat = {}
    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset()
        total_reward = 0

        episode_details, turn = get_details(env)

        for step in range(MAX_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = agent.actor(obs_tensor).squeeze(0)

            direction = map_action_to_direction(action)
            obs, reward, terminated, truncated, _ = env.step(direction)

            total_reward += reward
            if terminated or truncated:
                break

        result = '❌'
        if total_reward > 0.0:
            successes += 1
            # result = '✅'
        else:
            if turn in stat:
                stat[turn] += 1
            else:
                stat[turn] = 1
            # print(f"{result} Episode {ep}: Reward = {total_reward:.2f}, Steps = {step + 1}")
            # print(episode_details)

    print(f"Result: {successes}/{NUM_EPISODES} ({successes/NUM_EPISODES:.2f}%)")
    for turn in stat.keys():
        count = stat[turn]
        print(f"{turn}: {count}")

def get_details(env: DroneEnv):
    drone_x = env.drone_pos[0]
    drone_y = env.drone_pos[1]
    target_x = env.target_pos[0]
    target_y = env.target_pos[1]

    turn = direction_arrow(drone_x, drone_y, target_x, target_y)
    episode_details = f"\tDetails: " + \
                      f"drone [{drone_x:.2f}, {drone_y:.2f}]" + \
                      f" {turn} " + \
                      f"target [{target_x:.2f}, {target_y:.2f}]"
    return episode_details, turn


def direction_arrow(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    angle = math.degrees(math.atan2(dy, dx))  # [-180, 180]
    if angle < 0:
        angle += 360  # [0, 360)

    if 22.5 <= angle < 67.5:
        return "↗"  # up-right
    elif 67.5 <= angle < 112.5:
        return "↑"  # up
    elif 112.5 <= angle < 157.5:
        return "↖"  # up-left
    elif 157.5 <= angle < 202.5:
        return "←"  # left
    elif 202.5 <= angle < 247.5:
        return "↙"  # down-left
    elif 247.5 <= angle < 292.5:
        return "↓"  # down
    elif 292.5 <= angle < 337.5:
        return "↘"  # down-right
    else:
        return "→"  # right

if __name__ == "__main__":
    main()
