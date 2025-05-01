import torch

from agents.ddpg_agent import DDPGAgent
from env.drone_env import DroneEnv
from inference import get_last

DEVICE = "cpu"

def load_model():
    model_path = get_last.get_latest_model_path()
    env = DroneEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = 2  # [dx, dy] выход Actor-а

    agent = DDPGAgent(obs_dim, action_dim, device=DEVICE)

    # Загрузка весов
    checkpoint = torch.load(model_path, map_location=DEVICE)
    agent.actor.load_state_dict(checkpoint['actor'])
    print(f"🚀 Loaded model from {model_path}\n")
    agent.actor.eval()

    return env, agent