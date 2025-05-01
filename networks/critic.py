import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Выход: скалярная оценка действия
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)
