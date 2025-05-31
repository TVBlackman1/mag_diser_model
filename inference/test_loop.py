import math
import numpy as np
import torch

from env.drone_env import DroneEnv
from inference.load_model import load_model

DEVICE = "cpu"

NUM_EPISODES = 1000
MAX_STEPS = 300

def main():
    env, agent = load_model()

    successes = 0
    fails = 0
    stat = {}
    distance_percents = []
    for ep in range(1, NUM_EPISODES + 1):
        obs, _ = env.reset(options={'level_difficult': 'medium'})
        start_distance = env.last_distance_to_target
        for step in range(MAX_STEPS):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action = agent.actor(obs_tensor).squeeze(0)

            obs, reward, terminated, truncated, details = env.step(action)

            if terminated or truncated:
                if details['result'] == 'success':
                    successes += 1
                if details['result'] == 'fail':
                    fails += 1
                break
        end_distance = env.last_distance_to_target
        distance_percents.append(end_distance / start_distance)
    print(f"Result: {successes}/{NUM_EPISODES} ({successes/NUM_EPISODES:.2f}%)")
    print(f"Fails: {fails}/{NUM_EPISODES} ({fails/NUM_EPISODES:.2f}%)")
    ascii_histogram(distance_percents)
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

def ascii_histogram(values, bin_width=0.1, symbol="#", max_bar_width=50):
    if len(values) == 0:
        print("Empty input array.")
        return

    min_val = np.min(values)
    max_val = np.max(values)
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    counts, bin_edges = np.histogram(values, bins=bins)
    max_count = counts.max()

    print(f"Histogram (bin width = {bin_width}):\n")
    for count, edge_start, edge_end in zip(counts, bin_edges[:-1], bin_edges[1:]):
        bar_length = int((count / max_count) * max_bar_width)
        bar = symbol * bar_length
        print(f"{edge_start:6.2f} - {edge_end:6.2f}: {bar}")

if __name__ == "__main__":
    main()
