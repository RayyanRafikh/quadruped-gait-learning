import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class Config:
    def __init__(self):
        # Environment settings
        self.env_id = "MinitaurBulletEnv-v0"
        self.input_dims = [28]
        self.n_actions = 8
        
        # Training settings
        self.batch_size = 256
        self.num_episodes = 500
        self.gamma = 0.99
        self.tau = 0.005
        self.update_actor_interval = 2
        self.warmup_steps = 500
        self.replay_initial = 1000
        
        # Network settings
        self.hidden_size = 256
        self.learning_rate = 3e-4
        self.buffer_size = 100000

class Logger:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.scores = []
        self.avg_scores = []
        self.critic_losses = []
        self.actor_losses = []

    def log_step(self, episode, score, losses):
        self.scores.append(score)
        self.avg_scores.append(np.mean(self.scores[-100:]))
        
        if losses:
            self.critic_losses.append(losses.get('critic_loss', 0))
            self.actor_losses.append(losses.get('actor_loss', 0))

        if episode % 10 == 0:
            self.plot_results()
            self.save_results()

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.scores, label='Episode Reward')
        plt.plot(self.avg_scores, label='Average Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.savefig(f'{self.save_dir}/training_progress.png')
        plt.close()

    def save_results(self):
        results = {
            'scores': self.scores,
            'avg_scores': self.avg_scores,
            'critic_losses': self.critic_losses,
            'actor_losses': self.actor_losses
        }
        with open(f'{self.save_dir}/results.json', 'w') as f:
            json.dump(results, f)
