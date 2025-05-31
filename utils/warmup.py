import torch
from env.drone_env import DroneEnv

def run_warmup_on_env(agent, steps=500, level='warmup'):
    """
    Выполняет warm-up: записывает N переходов из одной среды в буфер агента.
    """
    env = DroneEnv()
    obs, _ = env.reset(options={'level_difficult': level})

    for _ in range(steps):
        move_direction = env.action_space.sample()
        action_tensor = torch.tensor(move_direction, dtype=torch.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
        done = terminated or truncated

        agent.replay_buffer.add(obs, action_tensor.numpy(), reward, next_obs, float(done))
        obs = next_obs if not done else env.reset()[0]

    print(f"✅ Warm-up: {steps} transitions collected from one env.")


def generate_warmup_experience(agent, num_envs=80, steps_per_env=120):
    """
    Создаёт num_envs сред (по одной за раз) и запускает warm-up на каждой.
    Все переходы направляются в один буфер агента.
    """
    total_transitions = 0

    for i in range(num_envs//3*2):
        print(f"▶ Warm-up env {i}...")
        run_warmup_on_env(agent, steps=steps_per_env, level='warmup')
        total_transitions += steps_per_env

    for i in range(num_envs//3*1):
        print(f"▶ Warm-up env {i}...")
        run_warmup_on_env(agent, steps=steps_per_env, level='warmup-obs')
        total_transitions += steps_per_env

    print(f"\n🏁 Warm-up complete: {total_transitions} transitions collected from {num_envs} environments.")
