# # utils/replay_buffer.py
#
# import random
# import numpy as np
# import torch
#
#
# class ReplayBuffer:
#     def __init__(self, buffer_size, batch_size, device):
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
#         self.device = device
#
#         self.memory = []
#         self.position = 0
#
#     def add(self, state, action, reward, next_state, done):
#         """Добавить один переход в буфер"""
#         if len(self.memory) < self.buffer_size:
#             self.memory.append(None)
#         self.memory[self.position] = (state, action, reward, next_state, done)
#         self.position = (self.position + 1) % self.buffer_size
#
#     def sample(self):
#         """Вернуть случайный батч для обучения"""
#         batch = random.sample(self.memory, self.batch_size)
#         states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
#
#         return (
#             torch.tensor(states, dtype=torch.float32).to(self.device),
#             torch.tensor(actions, dtype=torch.float32).to(self.device),
#             torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
#             torch.tensor(next_states, dtype=torch.float32).to(self.device),
#             torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
#         )
#
#     def __len__(self):
#         return len(self.memory)

# utils/replay_buffer.py

import random
import numpy as np
import torch

TOP_K = 4500 # Число лучших переходов для приоритизированного сэмплирования


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.memory = []
        self.mean = 0
        self.position = 0
        self.is_first_turn = True

    def add(self, state, action, reward, next_state, done):
        """Добавить один переход в буфер"""
        if len(self.memory) >= self.batch_size and reward < self.mean:
            return
        if self.position == 0 and len(self.memory) != 0:
            self.is_first_turn = False

        if len(self.memory) < self.buffer_size:
            self.memory.append(None)

        should_remove_from_mean = not self.is_first_turn
        if should_remove_from_mean:
            for_remove = self.memory[self.position]
            count = len(self.memory)
            count_after_removing = count - 1
            # removing
            self.mean = (self.mean * len(self.memory) - for_remove) / count_after_removing
            # adding
            self.mean = self.mean + (reward - self.mean) / count
        else:
            count = len(self.memory) + 1
            self.mean = self.mean + (reward - self.mean) / count

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        """Сэмплировать батч переходов из топ K лучших"""
        # Сортируем память по reward
        sorted_memory = sorted(self.memory, key=lambda x: x[2], reverse=True)

        # Берем только топ-K
        top_memory = sorted_memory[:min(TOP_K, len(sorted_memory))]

        # Выбираем случайный батч из топовых
        batch = random.sample(top_memory, min(self.batch_size, len(top_memory)))

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self.memory)