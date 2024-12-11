import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

class ActorNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.mu = nn.Linear(n_fc2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        return x

    def save(self, save_path):
        T.save(self.state_dict(), os.path.join(save_path, self.name + '_td3'))

    def load(self, load_path):
        self.load_state_dict(T.load(os.path.join(load_path, self.name + '_td3')))

class CriticNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.q = nn.Linear(n_fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        q = self.fc1(T.cat([state, action], dim=1))
        q = F.relu(q)
        q = self.fc2(q)
        q = F.relu(q)
        q = self.q(q)
        return q

    def save(self, save_path):
        T.save(self.state_dict(), os.path.join(save_path, self.name + '_td3'))

    def load(self, load_path):
        self.load_state_dict(T.load(os.path.join(load_path, self.name + '_td3')))
