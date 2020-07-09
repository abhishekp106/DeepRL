import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import gym
from random import sample
from random import random
from collections import namedtuple
import matplotlib.pyplot as plt

MEMORY_CAPACITY = 1000
NUM_EPISODES = 400
BATCH_SIZE = 32
DISCOUNT = 0.95
EPSILON_DECAY = 0.992

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 2)
    
    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
    
    def add(self, *args):
        self.contents.append(transition(*args))
        if len(self.contents) > self.capacity:
            self.contents.pop(0)

    def sample(self, num_samples):
        return sample(self.contents, num_samples)
    
    def length(self):
        return len(self.contents)
    
    def display(self):
        print(self.contents)

class DQN():
    def __init__(self, env, Q, num_actions, optimizer):
        self.env = env
        self.Q = Q
        self.Q_target = None
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.replay_memory = ReplayMemory(MEMORY_CAPACITY)
        self.loss_array = []
    
    def get_epsilon_greedy_action(self, s, exploration_prob):
        output = self.Q(torch.as_tensor(s, dtype=torch.float32))
        _, index = output.max(0)
        x = random()
        if x <= exploration_prob:
            # choose a random action that is not the best
            return sample([x for x in range(self.num_actions)], 1)[0]
        else:
            # choose the best action
            return index.item()
    
    def max_Qvalue(self, s):
        value, _ = self.Q(torch.as_tensor(s, dtype=torch.float32)).max(0)
        return value

    def preprocess_img(self, img):
        return img

    def experience_replay(self):
        if self.replay_memory.length() < MEMORY_CAPACITY:
            return 0
        transitions = self.replay_memory.sample(BATCH_SIZE)

        # create a mask to tell us how to calculate rewards
        non_terminal_mask = torch.tensor([not x.terminal for x in transitions], dtype=torch.int)
        batch_states = torch.tensor([x.state for x in transitions], dtype=torch.float32)
        batch_next_states = torch.tensor([x.next_state for x in transitions], dtype=torch.float32)
        batch_actions = torch.tensor([x.action for x in transitions], dtype=torch.int)
        batch_rewards = torch.tensor([x.reward for x in transitions], dtype=torch.float32)
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

        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted, target_values)
        self.loss_array.append(loss.item())

        # perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return q_values.mean()

    def train(self):
        EPSILON = 1.0
        self.env = env
        ep_rewards = []
        q_estimates = []
        ep_rewards_temp = []
        q_estimates_temp = []
        self.Q_target = MLP()
        self.Q_target.load_state_dict(self.Q.state_dict())
        num_steps = 1

        for episode in range(NUM_EPISODES):
            s = env.reset()
            done = False
            
            ep_reward = 0.0
            q_sum = 0.0
            ep_length = 0
            while not done:
                a = self.get_epsilon_greedy_action(s, EPSILON)
                s_new, reward, done, _ = env.step(a)
                self.replay_memory.add(s, a, reward, s_new, done)
                num_steps += 1
                ep_length += 1
                s = s_new
                ep_reward += reward
                x = self.experience_replay()
                q_sum += x
                if num_steps % 50 == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
            if EPSILON > 0.01:
                EPSILON *= EPSILON_DECAY
            
            ep_rewards.append(ep_reward)
            q_estimates.append((q_sum / ep_length))
            ep_rewards_temp.append(ep_reward)
            q_estimates_temp.append((q_sum / ep_length))
            if episode % 50 == 0:
                rewards = torch.as_tensor(ep_rewards_temp, dtype=torch.float)
                q = torch.as_tensor(q_estimates_temp, dtype=torch.float)
                print('Episode {} w/ Epsilon: {:.6}'.format(episode, EPSILON))
                print('Reward Mean: {:.3}, Std Dev: {:.3}, Max: {:.3}, Min: {:.3}'.format(rewards.mean().item(), rewards.std().item(), rewards.max().item(), rewards.min().item()))
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
    env = gym.make('CartPole-v0')
    Q = MLP()
    transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'terminal'])
    optimizer = torch.optim.Adam(Q.parameters())
    dqn = DQN(env, Q, 2, optimizer)
    dqn.train()