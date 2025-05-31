import torch
import torch.nn as nn
import torch.optim as optim

from networks.actor import Actor
from networks.critic import Critic
from utils.per_replay_buffer import PERReplayBuffer

class DDPGAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=int(1e6),
        batch_size=256,
        device="cpu",
        actor_update_period=5,
    ):
        self.actor_update_period = actor_update_period
        self.current_actor_update_index = 0
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.critic_loss_history = []
        self.actor_loss_history = []
        # Сети
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim, action_dim).to(device)

        # Таргетные сети
        self.target_actor = Actor(obs_dim, action_dim).to(device)
        self.target_critic = Critic(obs_dim, action_dim).to(device)

        # Копируем начальные веса
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Оптимизаторы
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = PERReplayBuffer(buffer_size, batch_size, device)

    @torch.no_grad()
    def select_action(self, obs, noise_std=0.1):
        """Выбираем действие на основе текущего состояния"""
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(obs).squeeze(0)

        # Добавляем небольшую стохастичность для исследования (exploration)
        noise = torch.randn_like(action) * noise_std
        action = torch.clamp(action + noise, -1, 1)

        return action

    def update(self):
        """Обновление нейронок на основе батча из буфера"""
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indexes = self.replay_buffer.sample()

        # Прямой проход через таргетные сети для следующего состояния
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Текущие Q-значения
        current_q = self.critic(states, actions)

        td_errors = (current_q - target_q.detach()).abs().view(-1)
        self.replay_buffer.update_priorities(indexes, td_errors.tolist())

        # Критик Loss
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Логгируем critic loss
        self.critic_loss_history.append(critic_loss.item())

        # Актор Loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_loss_history.append(actor_loss.item())

        self.current_actor_update_index += 1
        if self.current_actor_update_index == self.actor_update_period:
            self.current_actor_update_index = 0
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Обновляем таргетные сети (polyak-обновление)
            self.soft_update(self.target_actor, self.actor)
            self.soft_update(self.target_critic, self.critic)

    def soft_update(self, target_net, net):
        """Polyak обновление параметров"""
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict()
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
