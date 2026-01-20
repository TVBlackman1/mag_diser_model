import torch
import torch.nn as nn


class CriticAttention(nn.Module):
    def __init__(self, obs_dim_per_object, action_dim_total, n_heads=2, hidden_dim=128):
        super().__init__()
        self.obj_embed = nn.Linear(obs_dim_per_object, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + action_dim_total, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_objects, actions):
        # obs_objects: [batch, n_objects, obs_dim_per_object]
        # actions: [batch, action_dim_total]
        x = self.obj_embed(obs_objects)            # [batch, n_objects, hidden_dim]
        attn_out, _ = self.attn(x, x, x)          # self-attention
        pooled = attn_out.mean(dim=1)             # агрегированное представление
        x = torch.cat([pooled, actions], dim=-1)
        q_value = self.fc(x)
        return q_value