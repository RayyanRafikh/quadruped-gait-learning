import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.new_state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.terminal_buffer = np.zeros(self.buffer_size, dtype=np.bool_)

    def store(self, state, action, reward, state_, done):
        index = self.buffer_counter % self.buffer_size
        self.state_buffer[index] = state
        self.new_state_buffer[index] = state_
        self.terminal_buffer[index] = done
        self.reward_buffer[index] = reward
        self.action_buffer[index] = action
        self.buffer_counter += 1

    def sample_batch(self, batch_size):
        max_mem = min(self.buffer_counter, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_buffer[batch]
        states_new = self.new_state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        dones = self.terminal_buffer[batch]
        return states, actions, rewards, states_new, dones
