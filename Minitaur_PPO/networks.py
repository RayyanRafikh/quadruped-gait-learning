import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import numpy as np

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))
    
    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.logstd = AddBias(torch.zeros(n_actions))
        
        # Initialize weights
        for layer in self.pi:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state, noise=True):
        action_mean = self.pi(state)
        action_mean = torch.tanh(action_mean)
    
        logstd = self.logstd(torch.zeros_like(action_mean))
        logstd = torch.clamp(logstd, -20, 2)
    
        if noise:
            action_dist = distributions.Normal(action_mean, logstd.exp())
            action = action_dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = action_dist.log_prob(action).sum(-1)
        else:
            action = action_mean.detach()
            action_dist = None
            log_prob = None
        return action, log_prob, action_dist

    def evaluate(self, state, action):
        action_mean = self.pi(state)
        action_mean = torch.tanh(action_mean)
        logstd = self.logstd(torch.zeros_like(action_mean))
        logstd = torch.clamp(logstd, -20, 2)
        action_dist = distributions.Normal(action_mean, logstd.exp())
        return action_dist.log_prob(action).sum(-1), action_dist.entropy().sum(-1)

class Critic(nn.Module):
    def __init__(self, n_states, hidden_dim=256):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)

    def forward(self, state):
        return self.critic(state)
