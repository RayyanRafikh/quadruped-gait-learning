import os
import gym
import numpy as np
import torch
import pybullet_envs
import matplotlib.pyplot as plt
from datetime import datetime
from ppo import PPO
from config import Config

def plot_rewards(rewards, avg_rewards, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards')
    plt.plot(avg_rewards, label='Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'rewards.png'))
    plt.close()

def train():
    # Initialize config
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    # Initialize environment and agent
    env = gym.make("MinitaurBulletEnv-v0")
    eval_env = gym.make("MinitaurBulletEnv-v0")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    
    agent = PPO(n_states, n_actions, config)

    # Training tracking
    rewards = []
    avg_rewards = []
    best_avg_reward = float('-inf')
    timesteps_since_update = 0

    print(f"Starting training... Using device: {config.device}")
    
    # Training loop
    for episode in range(config.max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(config.max_timesteps):
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, _ = env.step(action)
            
            # Store experience
            agent.buffer.add(state, action, reward, done, log_prob, value)
            timesteps_since_update += 1
            
            state = next_state
            episode_reward += reward

            # Update if enough steps have been taken
            if timesteps_since_update >= config.update_timestep:
                value_loss, policy_loss, entropy_loss = agent.update()
                timesteps_since_update = 0
                
                print(f"Episode {episode} Update - Value Loss: {value_loss:.3f}, Policy Loss: {policy_loss:.3f}, Entropy Loss: {entropy_loss:.3f}")

            if done:
                break

        # Log episode results
        rewards.append(episode_reward)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        avg_rewards.append(avg_reward)

        print(f"Episode {episode} - Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

        # Save best model
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(os.path.join(config.model_dir, 'best_model.pth'))
            print(f"Saved new best model with average reward: {best_avg_reward:.2f}")

        # Evaluate agent periodically
        if episode % config.eval_interval == 0:
            eval_rewards = []
            for _ in range(10):  # 10 evaluation episodes
                state = eval_env.reset()
                eval_episode_reward = 0
                done = False
                
                while not done:
                    action, _, _ = agent.select_action(state)
                    state, reward, done, _ = eval_env.step(action)
                    eval_episode_reward += reward
                
                eval_rewards.append(eval_episode_reward)
            
            mean_eval_reward = np.mean(eval_rewards)
            print(f"\nEvaluation - Mean Reward: {mean_eval_reward:.2f}\n")
            
            # Save evaluation plot
            plot_rewards(rewards, avg_rewards, config.save_dir)

    # Final save
    agent.save(os.path.join(config.model_dir, 'final_model.pth'))
    plot_rewards(rewards, avg_rewards, config.save_dir)
    
    print("Training finished!")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
