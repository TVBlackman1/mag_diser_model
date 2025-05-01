import math
import numpy as np
import torch

class PERReplayBuffer:
    def __init__(self, buffer_size, batch_size, device, alpha=0.6, epsilon=1e-4):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha
        self.epsilon = epsilon

        self.memory = []
        self.priorities = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        priority = 1.0  # New transitions start with max priority
        if len(self.memory) < self.buffer_size:
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(priority)
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.buffer_size

    def sample(self):
        if len(self.memory) == 0:
            raise ValueError("Cannot sample from empty buffer.")

        priorities = np.array(self.priorities)
        scaled = (priorities + self.epsilon) ** self.alpha
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        batch = [self.memory[i] for i in indices]

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.float32).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device),
            list(indices)
        )

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = math.log1p(td)

    def get_stats(self):
        tds = np.array(self.priorities)
        return {
            "mean": np.mean(tds),
            "max": np.max(tds),
            "min": np.min(tds),
            "median": np.median(tds)
        }
    def __len__(self):
        return len(self.memory)
