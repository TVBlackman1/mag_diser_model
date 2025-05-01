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
from math import sqrt

TOP_K = 4500 # Число лучших переходов для приоритизированного сэмплирования


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.memory = []
        self.mean = 0
        self.std = 0
        self.M2 = 0  # сумма квадратов отклонений для std по методу Welford'а

        self.position = 0
        self.is_first_turn = True

    def add(self, state, action, reward, next_state, done):
        """Добавить один переход в буфер"""
        # if len(self.memory) >= self.batch_size and reward < self.mean:
        #     return
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
            mean_before = self.mean
            self.mean = (mean_before * len(self.memory) - for_remove) / count_after_removing
            delta = for_remove - mean_before
            self.M2 -= delta * (for_remove - self.mean)
            self.std = sqrt(self.M2 / (count_after_removing - 1))

            # adding
            delta = reward - self.mean
            self.mean += delta / count
            delta2 = reward - self.mean
            self.M2 += delta * delta2
            self.std = sqrt(self.M2 / (count - 1))
        else:
            count = len(self.memory) + 1
            delta = reward - self.mean
            self.mean += delta / count
            delta2 = reward - self.mean
            self.M2 += delta * delta2
            self.std = sqrt(self.M2 / (count-1))

        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        middle_ratio = 0.2
        bad_ratio = 0.2
        good_ratio = 0.6

        bottom_boundary = self.mean - self.std
        top_boundary = self.mean + self.std
        top = [x for x in self.memory if x[2] >= top_boundary]
        middle = [x for x in self.memory if bottom_boundary <= x[2] < top_boundary]
        bottom = [x for x in self.memory if x[2] < bottom_boundary]
        total_size = len(top) + len(middle) + len(bottom)

        n_top = int(total_size * good_ratio)
        n_middle = int(total_size * middle_ratio)
        n_bottom = int(total_size * bad_ratio)

        batch = [
            *random.sample(top, k=min(n_top, len(top))),
            *random.sample(bottom, k=min(n_bottom, len(bottom))),
            *random.sample(middle, k=min(n_middle, len(middle))),
        ]

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