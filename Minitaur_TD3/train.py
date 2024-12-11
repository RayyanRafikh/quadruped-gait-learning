import numpy as np
import torch as T
import gym
import argparse
import pybullet_envs
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
from networks import ActorNetwork, CriticNetwork
from buffer import ReplayBuffer
from agent import Agent

# Configuration constants
input_dims = [28]
n_actions = 8
batch_size = 256
env_id = "MinitaurBulletEnv-v0"
gamma = 0.99
tau = 0.005
update_actor = 2
replay_initial = 1000
warmup = 500
num_episodes = 500

def save_results(scores, avg_scores, actor_losses, critic_losses, save_dir):
    results = {
        'scores': scores,
        'average_scores': avg_scores,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses
    }
    
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in results.items()}, f)
    
    episodes = range(1, len(scores) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, scores, label='Episode Score')
    plt.plot(episodes, avg_scores, label='Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'scores.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, actor_losses, label='Actor Loss')
    plt.plot(episodes, critic_losses, label='Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

def choose_action(observation, act_net, noise, time_step, warmup=1000):
    if time_step < warmup:
        action = T.tensor(np.random.normal(scale=0.4, size=(n_actions,)))
    else:
        observation = T.tensor(observation, dtype=T.float).to(act_net.device)
        action = act_net(observation).to(act_net.device)
    action_ = T.tensor(action + noise, dtype=T.float).to(act_net.device)
    return T.clamp(action_, -1, 1).cpu().detach().numpy()

def save_models(actor, critic1, critic2, target_actor, target_critic1, target_critic2, save_path):
    print('... saving models ...')
    actor.save(save_path)
    critic1.save(save_path)
    critic2.save(save_path)
    target_actor.save(save_path)
    target_critic1.save(save_path)
    target_critic2.save(save_path)

def load_models(actor, critic1, critic2, target_actor, target_critic1, target_critic2, load_path):
    print('... loading models ...')
    actor.load(load_path)
    critic1.load(load_path)
    critic2.load(load_path)
    target_actor.load(load_path)
    target_critic1.load(load_path)
    target_critic2.load(load_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU", default=False, action='store_true', help='Enable CUDA True or False')
    parser.add_argument("--name", "--name", required=False, default='TD3_results', help="Name of the run")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.getcwd(), f'{args.name}_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    device = T.device("cuda:0" if args.GPU else "cpu")
    save_path = os.path.join(results_dir, 'models')
    os.makedirs(save_path, exist_ok=True)
    
    env = gym.make(env_id)
    
    # Initialize networks and agent
    act_net = ActorNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                          n_actions=8, name='actor')
    cri_net1 = CriticNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                            n_actions=8, name='critic1')
    cri_net2 = CriticNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                            n_actions=8, name='critic2')
    
    target_act = ActorNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                             n_actions=8, name='target_actor')
    target_cri1 = CriticNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                               n_actions=8, name='target_critic1')
    target_cri2 = CriticNetwork(lr=3e-4, input_dims=input_dims, n_fc1=256, n_fc2=256,
                               n_actions=8, name='target_critic2')
    
    replay_buffer = ReplayBuffer(buffer_size=100000, input_shape=[28], n_actions=8)
    noise = np.random.normal(scale=0.2)
    
    agent = Agent(actor=act_net, critic1=cri_net1, critic2=cri_net2,
             target_actor=target_act, target_critic1=target_cri1,
             target_critic2=target_cri2, replay_buffer=replay_buffer,
             noise=noise, tau=tau, gamma=gamma, batch_size=batch_size,
             update_actor=update_actor)
    
    scores = []
    avg_scores = []
    actor_losses = []
    critic_losses = []
    time_step = 0
    best_score = -1
    
    for i in range(num_episodes):
        obs = env.reset()
        finished = False
        score = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        n_updates = 0
        
        while not finished:
            act = choose_action(obs, act_net, noise, time_step=time_step, warmup=warmup)
            time_step += 1
            act = np.clip(act, -1, 1)
            state_, reward, finished, _ = env.step(act)
            
            agent.save_transition(obs, act, reward, state_, finished)
            obs = state_
            score += reward
            
            if replay_buffer.buffer_counter > max(batch_size, replay_initial):
                update_info = agent.update(replay_buffer)
                episode_actor_loss += update_info.get('actor_loss', 0)
                episode_critic_loss += update_info.get('critic_loss', 0)
                n_updates += 1
        
        # Calculate average losses for the episode
        if n_updates > 0:
            episode_actor_loss /= n_updates
            episode_critic_loss /= n_updates
        
        scores.append(score)
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if score > best_score:
            best_score = score
            save_models(act_net, cri_net1, cri_net2, target_act, target_cri1, target_cri2, save_path)
        
        print(f'Episode: {i}, Score: {score:.2f}, Average Score: {avg_score:.2f}')
        
        if i % 10 == 0:
            save_results(scores, avg_scores, actor_losses, critic_losses, results_dir)
    
    save_results(scores, avg_scores, actor_losses, critic_losses, results_dir)
    env.close()