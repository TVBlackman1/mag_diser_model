import torch
import torch.nn as nn

# Параметры
batch_size = 2
hidden_dim = 32
num_heads = 4
n_drones = 2
n_obstacles = 2
n_target = 1
obs_dim_drone = 4
action_dim = 2
obs_dim_obstacle = 3
obs_dim_target = 2

# Dummy данные
drone_obs = torch.rand(batch_size, n_drones, obs_dim_drone)
drone_actions = torch.rand(batch_size, n_drones, action_dim)

obstacles = torch.rand(batch_size, n_obstacles, obs_dim_obstacle)
target = torch.rand(batch_size, n_target, obs_dim_target)

# Embeddings для токенов
drone_embed = nn.Linear(obs_dim_drone + action_dim, hidden_dim)
obstacle_embed = nn.Linear(obs_dim_obstacle, hidden_dim)
target_embed = nn.Linear(obs_dim_target, hidden_dim)

# Преобразуем объекты в токены
drone_tokens = drone_embed(torch.cat([drone_obs, drone_actions], dim=-1))
obstacle_tokens = obstacle_embed(obstacles)
target_tokens = target_embed(target)

# Собираем все токены
all_tokens = torch.cat([drone_tokens, target_tokens, obstacle_tokens], dim=1)  # [batch, n_tokens, hidden_dim]

# Attention
attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
attn_out, _ = attn(all_tokens, all_tokens, all_tokens)  # self-attention

# Финальный MLP критика
critic_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, 1)  # Q-value
)
q_values = critic_head(attn_out.mean(dim=1))  # глобальное состояние

print("Q-values:", q_values)
print("Shape:", q_values.shape)