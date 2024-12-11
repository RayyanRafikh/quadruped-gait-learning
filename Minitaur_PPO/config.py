from dataclasses import dataclass
import torch
import os
from datetime import datetime

@dataclass
class Config:
    # Training parameters
    max_episodes: int = 1000
    max_timesteps: int = 5000
    update_timestep: int = 4096
    eval_interval: int = 100
    
    # PPO parameters
    hidden_dim: int = 256
    lr: float = 2e-4
    gamma: float = 0.95
    gae_lambda: float = 0.99
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    num_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 100000
    max_grad_norm: float = 0.5
    
    # Device and logging
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir: str = os.path.join('results', f'minitaur_ppo_{timestamp}')
    model_dir: str = os.path.join(save_dir, 'models')
