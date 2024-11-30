## Work in Progress
from collections import deque
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import patches
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class Plotter:
    def __init__(self, trial_data):
        self.trial_data = trial_data

    def plot_env(self, env, ax):
        height, width = env.shape
        ax.imshow(1 - env, cmap=cm.binary, alpha=0.25, zorder=1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim([0, width - 1])
        ax.set_ylim([height - 1, 0])
        return ax

    def plot_prey(self, prey_location, ax):
        ax.scatter(prey_location[1], prey_location[0], c='Black', s=100)
        return ax

    def plot_episode(self, episode_data, ax):
        env = np.array(episode_data['env'])
        agent_path = np.array(episode_data['path'])
        prey_location = episode_data['prey_location'][0]
        
        ax = self.plot_env(env, ax)
        ax = self.plot_prey(prey_location, ax)
        ax.plot(agent_path[:, 1], agent_path[:, 0], color='Grey')
            
        ax.set_title(f'{len(agent_path) - 1} Steps')
        return ax

    def plot_reward(self):
        rewards = [self.trial_data[trial]['rewards'] for trial in range(len(self.trial_data))]
        n = len(rewards)
        
        smoothed_reward = [np.mean(rewards[max(0, i - 100) : i + 1]) for i in range(n)]
        plt.scatter(np.arange(n), rewards, linewidth=0, alpha=0.5, c='C0', label='Reward')
        plt.plot(np.arange(len(smoothed_reward)), smoothed_reward, color='k', linestyle='--', linewidth=0.5, label='Smoothed')
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend()
        plt.title('Training Reward')
        return plt

    def plot_paths(self):
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
            
        for i in range(5):
            trial_first = self.trial_data[i]
            ax_first = axs[0, i]
            self.plot_episode(trial_first, ax=ax_first)
              
        for i in range(5):
            trial_last = self.trial_data[len(self.trial_data) - (i + 1)]
            ax_last = axs[1, i]
            self.plot_episode(trial_last, ax=ax_last)

    def animated_trial(self, trial):
        data = self.trial_data[trial]
        env = np.array(data['env'])
        fig, ax = plt.subplots()
        ax = self.plot_env(env, ax)

        agent_animation = ax.scatter([], [], s=120, color='k', zorder=4)
        prey_animation = ax.scatter([], [], s=60, color='k', alpha=0.5, marker='o', zorder=3)

        ax.imshow(1 - env[:, :], cmap='binary', interpolation='nearest', alpha=0.2, zorder=1)

        def update_animation(t):
            agent_location = data['path'][t]
            agent_animation.set_offsets([agent_location[1], agent_location[0]])

            prey_location = data['prey_location'][t]
            prey_animation.set_offsets([prey_location[1], prey_location[0]])

        frames = len(data['path'])
        anim = animation.FuncAnimation(ax.figure, update_animation, frames=frames, interval=200, blit=False)
        anim.save("Trial.gif", dpi=300, writer='pillow')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc_out(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_out = nn.Linear(128, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc_out(x)
        return q_value

class ReplayBuffer:
    def __init__(self, max_size=1e6):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = map(
            np.stack, zip(*batch)
        )
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        self.replay_buffer = ReplayBuffer()
        
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005
        
        self.max_action = max_action
        self.noise_std_dev = 0.1 * max_action

    def select_action(self, state, noise=True):
        state_tensor = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if noise:
            noise = np.random.normal(0, self.noise_std_dev, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)

        return action
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state_batch)
        action = torch.FloatTensor(action_batch)
        reward = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state = torch.FloatTensor(next_state_batch)
        done = torch.FloatTensor(done_batch).unsqueeze(1)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)

    def _soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
