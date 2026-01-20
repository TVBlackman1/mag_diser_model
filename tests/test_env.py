# test_env.py
from env.drone_env import DroneEnv
from utils.generation import EnvGeneratorDifferentEpisodes
from config.env_config import FIELD_SIZE
from config.train_config import NUM_EPISODES, MAX_STEPS_PER_EPISODE

def main():
    generator = EnvGeneratorDifferentEpisodes(FIELD_SIZE, NUM_EPISODES, MAX_STEPS_PER_EPISODE)
    env = DroneEnv(generator)
    obs, info = env.reset()

    print("=== Environment Reset ===")
    env.render()

    done = False
    step_count = 0

    while not done and step_count < 20:
        action = env.action_space.sample()  # случайное действие (continuous)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step_count + 1}")
        print(f"Action taken: {action}")
        print(f"Observation: {obs}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        env.render()

        done = terminated or truncated
        step_count += 1

    print("\n=== Episode finished ===")

if __name__ == "__main__":
    main()
