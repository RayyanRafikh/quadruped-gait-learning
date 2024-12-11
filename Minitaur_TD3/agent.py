import torch as T
import torch.nn.functional as F
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
import numpy as np

class Agent:
    def __init__(self, actor, critic1, critic2, target_actor, target_critic1, 
             target_critic2, replay_buffer, noise, tau, gamma, batch_size, update_actor):
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2
        self.target_actor = target_actor
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.tau = tau
        self.gamma = gamma
        self.noise = noise
        self.update_actor = update_actor

    def save_transition(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, reward, new_state, done)

    def update(self, buffer):
        if buffer.buffer_counter < self.batch_size:
            return {}

        state, action, reward, new_state, done = buffer.sample_batch(self.batch_size)
    
        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)

        target_actions = self.target_actor(new_state)
        target_actions = target_actions + T.clamp(
            T.randn_like(target_actions) * 0.2, -0.5, 0.5)
        target_actions = T.clamp(target_actions, -1, 1)

        q1_ = self.target_critic1(new_state, target_actions)
        q2_ = self.target_critic2(new_state, target_actions)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        critic_value = T.min(q1_, q2_)
        reward = reward.view(-1, 1)
        target = reward + self.gamma * critic_value

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        actor_loss = 0
        if buffer.buffer_counter % self.update_actor == 0:
            self.actor.optimizer.zero_grad()
            actor_q1_loss = self.critic1(state, self.actor(state))
            actor_loss = -T.mean(actor_q1_loss)
            actor_loss.backward()
            self.actor.optimizer.step()

            self._update_networks()

        return {
            'actor_loss': float(actor_loss),
            'critic_loss': float(critic_loss)
        }

    def _update_networks(self):
        actor_params = dict(self.actor.named_parameters())
        critic1_params = dict(self.critic1.named_parameters())
        critic2_params = dict(self.critic2.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        target_critic_params1 = dict(self.target_critic1.named_parameters())
        target_critic_params2 = dict(self.target_critic2.named_parameters())

        for name in critic1_params:
            critic1_params[name] = self.tau * critic1_params[name].clone() + \
                                 (1-self.tau) * target_critic_params1[name].clone()

        for name in critic2_params:
            critic2_params[name] = self.tau * critic2_params[name].clone() + \
                                 (1-self.tau) * target_critic_params2[name].clone()

        for name in actor_params:
            actor_params[name] = self.tau * actor_params[name].clone() + \
                               (1-self.tau) * target_actor_params[name].clone()

        self.target_critic1.load_state_dict(critic1_params)
        self.target_critic2.load_state_dict(critic2_params)
        self.target_actor.load_state_dict(actor_params)
