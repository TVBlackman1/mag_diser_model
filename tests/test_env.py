# test_env.py
from agents.ddpg_agent import DIRECTIONS
from env.drone_env import DroneEnv

def main():
    env = DroneEnv()
    obs, info = env.reset()

    print("=== Environment Reset ===")
    env.render()

    done = False
    step_count = 0

    while not done and step_count < 20:
        action_idx = env.action_space.sample()  # случайное действие
        direction = DIRECTIONS[action_idx]
        obs, reward, terminated, truncated, info = env.step(direction)

        print(f"\nStep {step_count + 1}")
        print(f"Action taken: {direction}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        env.render()

        done = terminated or truncated
        step_count += 1

    print("\n=== Episode finished ===")

if __name__ == "__main__":
    main()
