# tests/test_env_obs.py
from env.drone_env import DroneEnv
from utils.generation import EnvGeneratorDifferentEpisodes
from config.env_config import FIELD_SIZE
from config.train_config import NUM_EPISODES, MAX_STEPS_PER_EPISODE
import numpy as np

def main():
    generator = EnvGeneratorDifferentEpisodes(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    env = DroneEnv(generator)
    obs, _ = env.reset()

    distance_history = []

    print("=== Initial observation ===")
    print(f"obs = {obs}")
    print(f"cos(theta) = {obs[0]:.4f}, sin(theta) = {obs[1]:.4f}, exist_flag = {obs[2]}, normalized_distance = {obs[3]:.4f}")
    distance_history.append(obs[3])  # Сохраняем первую дистанцию
    print("---------------------------")

    for i in range(30):
        action = env.action_space.sample()  # continuous action
        obs, reward, terminated, truncated, info = env.step(action)

        distance_normalized = obs[3]
        distance_history.append(distance_normalized)

        print(f"Step {i+1}")
        print(f"Action: {action}")
        print(f"cos(theta) = {obs[0]:.4f}, sin(theta) = {obs[1]:.4f}, exist_flag = {obs[2]}, normalized_distance = {obs[3]:.4f}")
        print(f"Reward: {reward:.2f}")
        print("---------------------------")

        if terminated or truncated:
            print("Episode ended.")
            break

    # Выведем как менялось расстояние до цели по шагам
    print("\n=== Distance to target over steps ===")
    for step_idx, dist in enumerate(distance_history):
        print(f"Step {step_idx}: distance_normalized = {dist:.4f}")

if __name__ == "__main__":
    main()
