# tests/test_env_obs.py

from env.drone_env import DroneEnv
import numpy as np

def main():
    env = DroneEnv()
    obs, _ = env.reset()

    distance_history = []

    print("=== Initial observation ===")
    print(f"obs = {obs}")
    print(f"cos(theta) = {obs[0]:.4f}, sin(theta) = {obs[1]:.4f}, exist_flag = {obs[2]}, normalized_distance = {obs[3]:.4f}")
    distance_history.append(obs[3])  # Сохраняем первую дистанцию
    print("---------------------------")

    for i in range(30):
        action = env.action_space.sample()
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
