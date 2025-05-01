import random
import numpy as np
import torch
from math import sqrt


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.memory = []
        self.priorities = [] # TD-error priority
        self.mean = 0
        self.std = 0
        self.M2 = 0  # сумма квадратов отклонений для std по методу Welford'а

        self.position = 0
        self.is_first_turn = True

    def add(self, state, action, reward, next_state, done):
        """Добавить один переход в буфер"""

        if self.position == 0 and len(self.memory) != 0:
            self.is_first_turn = False

        if len(self.memory) < self.buffer_size:
            self.memory.append(None)
            self.priorities.append(1.0) # max by default

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

        priorities_np = np.array(self.priorities)
        mean = priorities_np.mean()
        std = priorities_np.std()
        bottom_boundary = mean - std
        top_boundary = mean + std

        # определяем группы индексов
        high_idx = [i for i, p in enumerate(priorities_np) if p >= top_boundary]
        mid_idx = [i for i, p in enumerate(priorities_np) if bottom_boundary <= p < top_boundary]
        low_idx = [i for i, p in enumerate(priorities_np) if p < bottom_boundary]

        n_high = int(self.batch_size * good_ratio)
        n_mid = int(self.batch_size * middle_ratio)
        n_low = self.batch_size - n_high - n_mid

        sampled_indices = (
                random.sample(high_idx, min(n_high, len(high_idx))) +
                random.sample(mid_idx, min(n_mid, len(mid_idx))) +
                random.sample(low_idx, min(n_low, len(low_idx)))
        )

        # перемешиваем финальный набор индексов
        random.shuffle(sampled_indices)

        # формируем батч по этим индексам
        batch = [self.memory[i] for i in sampled_indices]
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device),
            sampled_indices
        )

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)