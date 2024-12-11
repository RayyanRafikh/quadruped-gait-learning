import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from networks import Actor, Critic
from buffer import RolloutBuffer

class PPO:
    def __init__(
        self,
        n_states,
        n_actions,
        config
    ):
        self.device = config.device
        
        # Initialize networks
        self.actor = Actor(n_states, n_actions, config.hidden_dim).to(self.device)
        self.critic = Critic(n_states, config.hidden_dim).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam([
            *self.actor.parameters(),
            *self.critic.parameters()
        ], lr=config.lr)
        
        # Initialize rollout buffer
        self.buffer = RolloutBuffer(config.buffer_size, n_states, n_actions, self.device)
        
        # Save parameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_param = config.clip_param
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        
        # Setup logging
        self.writer = SummaryWriter(config.save_dir)
        self.total_steps = 0
        
    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _ = self.actor(state)
            value = self.critic(state)
            
        return (
            action.cpu().numpy()[0],
            log_prob.cpu().numpy(),
            value.cpu().numpy()[0]
        )
    
    def update(self):
        # Compute returns and advantages
        with torch.no_grad():
            last_value = self.critic(
                torch.FloatTensor(self.buffer.states[self.buffer.ptr-1]).unsqueeze(0).to(self.device)
            ).cpu().numpy()[0]
        
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # PPO update
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy_loss = 0
        
        for _ in range(self.num_epochs):
            states, actions, old_log_probs, returns, advantages = self.buffer.get()
            
            for idx in range(0, states.size(0), self.batch_size):
                # Get mini-batch
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_old_log_probs = old_log_probs[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                new_values = self.critic(batch_states)
                
                # Policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [*self.actor.parameters(), *self.critic.parameters()],
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        # Log statistics
        num_updates = self.num_epochs * (self.buffer.size // self.batch_size)
        avg_value_loss = total_value_loss / num_updates
        avg_policy_loss = total_policy_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates
        
        self.writer.add_scalar('Loss/Value', avg_value_loss, self.total_steps)
        self.writer.add_scalar('Loss/Policy', avg_policy_loss, self.total_steps)
        self.writer.add_scalar('Loss/Entropy', avg_entropy_loss, self.total_steps)
        
        # Clear buffer
        self.buffer.clear()
        
        return avg_value_loss, avg_policy_loss, avg_entropy_loss
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
