import os
import gym
import numpy as np
import torch
import pybullet_envs
import pybullet as p
import time
from ppo import PPO
from config import Config

def test(model_path, render=True, n_episodes=10):
    # Initialize config and environment
    config = Config()
    env = gym.make("MinitaurBulletEnv-v0", render=render)
    
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    # Initialize agent
    agent = PPO(n_states, n_actions, config)
    agent.load(model_path)

    # Setup camera if rendering
    if render:
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.5]
        )

    # Testing loop
    test_rewards = []
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            if render:
                time.sleep(1/60)  # For visualization
            
            # Select action without noise
            with torch.no_grad():
                action, _, _ = agent.select_action(state)
            
            # Execute action
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            step += 1

        test_rewards.append(episode_reward)
        print(f'Episode {episode + 1} - Reward: {episode_reward:.2f}, Steps: {step}')

    # Print summary
    mean_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    print(f'\nResults over {n_episodes} episodes:')
    print(f'Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}')
    
    env.close()

if __name__ == "__main__":
    # Specify the path to your trained model
    model_path = 'results/minitaur_ppo_20241211_051309/models/best_model.pth'
    test(model_path, render=True, n_episodes=10)
