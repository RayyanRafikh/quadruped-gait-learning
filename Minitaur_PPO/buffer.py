import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, n_states, n_actions, device):
        self.states = np.zeros((buffer_size, n_states), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
        self.buffer_size = buffer_size
        self.device = device

    def add(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        last_gae_lam = 0
        for step in reversed(range(self.size)):
            if step == self.size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std

    def get(self):
        indices = np.random.permutation(self.size)
        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.log_probs[indices]).to(self.device),
            torch.FloatTensor(self.returns[indices]).to(self.device),
            torch.FloatTensor(self.advantages[indices]).to(self.device)
        )

    def clear(self):
        self.ptr = 0
        self.size = 0
