import torch
from env.drone_env import DroneEnv
from utils.generation import EnvGeneratorDifferentEpisodes, EnvGeneratorDynamic

def run_warmup_on_env(agent, episode: int, field_size, steps=500, level='warmup'):
    """
    Выполняет warm-up: записывает N переходов из одной среды в буфер агента.
    """
    
    env_generator = EnvGeneratorDynamic(field_size, 10, steps)
    env = DroneEnv(env_generator)

    for step in range(steps):
        env_generator.set_state(episode=episode, step=step, difficult_level=level)
        obs, _ = env.reset()
        move_direction = env.action_space.sample()
        action_tensor = torch.tensor(move_direction, dtype=torch.float32)

        next_obs, reward, terminated, truncated, _ = env.step(action_tensor)
        done = terminated or truncated

        agent.replay_buffer.add(obs, action_tensor.numpy(), reward, next_obs, float(done))
        obs = next_obs if not done else env.reset()[0]


def generate_warmup_experience(agent, field_size=10, num_envs=80, steps_per_env=120):
    """
    Создаёт num_envs сред (по одной за раз) и запускает warm-up на каждой.
    Все переходы направляются в один буфер агента.
    """
    total_transitions = 0

    episode_number = 0
    for i in range(num_envs//3*2):
        run_warmup_on_env(agent, episode_number, field_size, steps=steps_per_env, level='warmup')
        episode_number += 1
        total_transitions += steps_per_env

    for i in range(num_envs//3*1):
        run_warmup_on_env(agent, episode_number, field_size, steps=steps_per_env, level='warmup-obs')
        episode_number += 1
        total_transitions += steps_per_env

    print(f"\n🏁 Warm-up complete: {total_transitions} transitions collected from {num_envs} environments.")
