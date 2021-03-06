from random import random
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import torch.tensor as tensor
import gym
import matplotlib.pyplot as plt
import numpy as np
import gc

import wrappers
from model import CNN
from replay_memory import ReplayMemory

MEMORY_CAPACITY = 10000
NUM_EPISODES = 20000
BATCH_SIZE = 32
DISCOUNT = 0.99
TARGET_UPDATE = 1000
PATH = './dqn_atari_net.pth'

class DQN():
    def __init__(self, env, Q, num_actions, optimizer):
        self.env = env
        self.Q = Q
        self.Q_target = None
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.replay_memory = ReplayMemory(MEMORY_CAPACITY)
        self.loss_array = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_epsilon_greedy_action(self, q_values, exploration_prob):
        x = random()
        if x <= exploration_prob:
            # choose a random action that is not the best
            return self.env.action_space.sample()
        else:
            # choose the best (predicted) action
            return np.argmax(q_values[0])
    
    def max_Qvalue(self, s):
        value, _ = self.Q(torch.as_tensor(s, dtype=torch.float32)).unsqueeze(1).max(0)
        return value

    def get_q_values(self, s):
        with torch.no_grad():
            state = torch.tensor(s, dtype=torch.float32).detach().to(self.device).unsqueeze(0)
            q_values = self.Q(state)
            return q_values.to('cpu').detach().numpy()

    def get_state(self, obs):
        #state = state.transpose((2, 0, 1))
        return torch.from_numpy(obs).unsqueeze(0).detach()

    def experience_replay(self, num_steps):
        if self.replay_memory.length < BATCH_SIZE or num_steps <= MEMORY_CAPACITY:
            return -1
        transitions = self.replay_memory.sample(BATCH_SIZE)

        # create a mask to tell us how to calculate rewards
        non_terminal_mask = torch.tensor([not x.terminal for x in transitions], dtype=torch.int, device=self.device)
        batch_states = torch.from_numpy(np.stack([x.state for x in transitions])).float().to(self.device)
        #print(np.stack([x.state for x in transitions]).shape)
        batch_next_states = torch.from_numpy(np.stack([x.state for x in transitions])).float().to(self.device)
        #print(batch_states.shape)
        batch_actions = torch.tensor([x.action for x in transitions], dtype=torch.int, device=self.device)
        batch_rewards = torch.tensor([x.reward for x in transitions], dtype=torch.float32, device=self.device)
        output = self.Q(batch_states)
        
        # we SELECT the Q-values of the actions that we took for each transition
        l = []
        for i, row in enumerate(output):
            l.append(row[batch_actions[i]])
        predicted = torch.stack(l, 0)

        # Calculate target values y + max(Q(s_new)), using our mask from earlier
        q_values = self.Q_target(batch_next_states)
        next_state_values = q_values.max(1)[0].detach()
        target_values = batch_rewards + (next_state_values * DISCOUNT) * (non_terminal_mask)

        criterion = nn.MSELoss()
        loss = criterion(predicted, target_values)
        self.loss_array.append(loss.item())

        # perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del batch_states
        del batch_next_states
        torch.cuda.empty_cache()

        # comment out if using GPU
        return q_values.mean().detach().item()

    def train(self):
        EPSILON = 1.0
        EPSILON_MIN = 0.05
        EPSILON_STEP = (EPSILON - EPSILON_MIN) / (1000000)
        self.env = env
        ep_rewards = []
        q_estimates = []
        ep_rewards_temp = []
        q_estimates_temp = []
        self.Q_target = CNN((84, 84), 4).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        num_steps = 1
        displayed = False

        for episode in range(NUM_EPISODES):
            s = env.reset()
            #s = self.get_state(s)
            done = False
            
            ep_reward = 0.0
            q_sum = 0.0
            ep_length = 0
            while not done:
                q_values = self.get_q_values(s)
                a = self.get_epsilon_greedy_action(q_values, EPSILON)
                s_new, reward, done, _ = env.step(a)

                self.replay_memory.add(s, a, reward, s_new, done)
                if self.replay_memory.length >= MEMORY_CAPACITY and not displayed:
                    print('Capacity reached. *********************************************')
                    displayed = True
                num_steps += 1
                ep_length += 1
                s = s_new
                ep_reward += reward
                x = self.experience_replay(num_steps)
                q_sum += x
                if EPSILON > EPSILON_MIN and num_steps >= MEMORY_CAPACITY:
                    EPSILON -= EPSILON_STEP
                if num_steps % TARGET_UPDATE == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                if num_steps % 4000 == 0:
                    torch.save(self.Q.state_dict(), PATH)
            
            
            #ep_rewards.append(ep_reward)
            #q_estimates.append((q_sum / ep_length))
            ep_rewards_temp.append(ep_reward)
            q_estimates_temp.append((q_sum / ep_length))
            if episode % 50 == 0:
                rewards = torch.as_tensor(ep_rewards_temp, dtype=torch.float).detach()
                q = torch.as_tensor(q_estimates_temp, dtype=torch.float).detach()
                print('Episode {} w/ Epsilon: {:.6}'.format(episode, EPSILON))
                print('Reward Sum: {:.3}, Mean: {:.3}, Std Dev: {:.3}, Max: {:.3}, Min: {:.3}'.format(rewards.sum().item(), rewards.mean().item(), rewards.std().item(), rewards.max().item(), rewards.min().item()))
                print('Q-Value Mean: {:.3}, Std Dev: {:.3}, Max: {:.3}, Min: {:.3}'.format(q.mean().item(), q.std().item(), q.max().item(), q.min().item()))
                ep_rewards_temp = []
                q_estimates_temp = []
        
        #%%
        plt.plot(ep_rewards)
        plt.show()

        #%%
        plt.plot(q_estimates)
        plt.show()

        #%%
        plt.plot(self.loss_array)
        plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('BreakoutNoFrameskip-v4')
    env = wrappers.Create(env)
    print(env.action_space)
    Q = CNN((84, 84), 4).to(device)
    #transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'terminal'])
    optimizer = torch.optim.Adam(Q.parameters(), lr=1e-4)
    dqn = DQN(env, Q, 2, optimizer)
    dqn.train()