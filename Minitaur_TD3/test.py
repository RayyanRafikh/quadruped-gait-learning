# test.py
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import gym
import pybullet_envs
import pybullet as p
import argparse
import os
import time

class TestNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2, n_actions, name):
        super(TestNetwork, self).__init__()
        self.input_dims = input_dims
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, n_fc1)
        self.fc2 = nn.Linear(n_fc1, n_fc2)
        self.mu = nn.Linear(n_fc2, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))
        return x

    def load(self, load_path):
        print('... loading model ...')
        self.load_state_dict(T.load(load_path, map_location=self.device))

def test_agent(model, env, n_episodes=10, render=True):
    scores = []
    
    if render:
        p.resetDebugVisualizerCamera(
            cameraDistance=2.0,
            cameraYaw=90,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 1]
        )

    for episode in range(n_episodes):
        score = 0
        done = False
        obs = env.reset()

        while not done:
            if render:
                env.render()
                time.sleep(1./60.)  # 60 FPS rendering

            # Convert observation to tensor
            state = T.tensor(obs, dtype=T.float).to(model.device)
            
            # Get action from model
            with T.no_grad():
                action = model(state).cpu().numpy()
            
            # Take step in environment
            obs, reward, done, _ = env.step(action)
            score += reward

        scores.append(score)
        print(f'Episode {episode+1}: Score = {score:.2f}')

    avg_score = np.mean(scores)
    print(f'\nAverage Score over {n_episodes} episodes: {avg_score:.2f}')
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to test")
    parser.add_argument("--no_render", action='store_true', help="Disable rendering")
    parser.add_argument("--GPU", action='store_true', help="Use GPU if available")
    args = parser.parse_args()

    # Environment and model parameters
    env_id = "MinitaurBulletEnv-v0"
    input_dims = [28]
    n_actions = 8

    # Create environment
    env = gym.make(env_id, render=not args.no_render)

    # Initialize model
    model = TestNetwork(
        lr=0.0,  # Not needed for testing
        input_dims=input_dims,
        n_fc1=256,
        n_fc2=256,
        n_actions=n_actions,
        name='test_actor'
    )

    # Load trained model
    model.load(args.model_path)
    
    try:
        # Test the agent
        scores = test_agent(
            model=model,
            env=env,
            n_episodes=args.episodes,
            render=not args.no_render
        )
        
    finally:
        env.close()

