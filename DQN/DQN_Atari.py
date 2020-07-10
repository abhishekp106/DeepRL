import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import gym
from random import sample
from random import random
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import PIL.ImageOps as imageops
from PIL import Image
from skimage.transform import downscale_local_mean
import wrappers

MEMORY_CAPACITY = 10000
NUM_EPISODES = 400
BATCH_SIZE = 32
DISCOUNT = 0.99
EPSILON_DECAY = 0.99
TARGET_UPDATE = 1000

class CNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNN, self).__init__()
        # input is (100, 80)
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        y_output, x_output = self.get_output_size((input_shape[0], input_shape[1]), 8, 4)
        y_output, x_output = self.get_output_size((y_output, x_output), 4, 2)
        y_output, x_output = self.get_output_size((y_output, x_output), 3, 1)
        #print((y_output, x_output))
        self.num_features = 64 * (y_output * x_output)
        self.fc1 = nn.Linear(self.num_features, 100)
        self.fc2 = nn.Linear(100, num_actions)
    
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.num_features)
        
        #print(x.shape)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def get_output_size(self, input_shape, field_size, stride):
        y, x = input_shape
        return int(((y - field_size) / stride) + 1), int(((x - field_size) / stride) + 1)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []
        self.idx = 0
    
    def add(self, s, a, r, s_new, terminal):
        if self.idx < self.capacity:
            self.contents.append(transition(s, a, r, s_new, terminal))
        else:
            self.contents[self.idx] = transition(s, a, r, s_new, terminal)
        self.idx = (self.idx + 1) % self.capacity

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
        x = random()
        if x <= exploration_prob:
            # choose a random action that is not the best
            return self.env.action_space.sample()
        else:
            # choose the best (predicted) action
            with torch.no_grad():
                _, index = self.Q(s)[0].max(0)
                return index.item()
    
    def max_Qvalue(self, s):
        value, _ = self.Q(torch.as_tensor(s, dtype=torch.float32)).unsqueeze(1).max(0)
        return value

    def process_img(self, img):
        grayscaled_img = imageops.grayscale(Image.fromarray(img))
        downscaled_img = Image.fromarray(downscale_local_mean(np.array(grayscaled_img), (2,2)))
        return np.array(imageops.grayscale(downscaled_img))[:100, :]

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0).detach()

    def experience_replay(self):
        if self.replay_memory.length() < BATCH_SIZE:
            return -1
        transitions = self.replay_memory.sample(BATCH_SIZE)

        # create a mask to tell us how to calculate rewards
        non_terminal_mask = torch.tensor([not x.terminal for x in transitions], dtype=torch.int, device=device)
        batch_states = torch.cat([x.state for x in transitions], dim=0)
        batch_next_states = torch.cat([x.next_state for x in transitions], dim=0)
        batch_actions = torch.tensor([x.action for x in transitions], dtype=torch.int, device=device)
        batch_rewards = torch.tensor([x.reward for x in transitions], dtype=torch.float32, device=device)
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

        # comment out if using GPU
        return q_values.mean()

    def train(self):
        EPSILON = 1.0
        self.env = env
        ep_rewards = []
        q_estimates = []
        ep_rewards_temp = []
        q_estimates_temp = []
        self.Q_target = CNN((84, 84), 6).to(device)
        self.Q_target.load_state_dict(self.Q.state_dict())
        num_steps = 1

        for episode in range(NUM_EPISODES):
            s = env.reset()
            print(s.shape)
            s = self.get_state(s)
            print(s.shape)
            done = False
            
            ep_reward = 0.0
            q_sum = 0.0
            ep_length = 0
            while not done:
                a = self.get_epsilon_greedy_action(s, EPSILON)
                s_new, reward, done, _ = env.step(a)
                s_new = self.get_state(s_new)

                self.replay_memory.add(s, a, reward, s_new, done)
                if (self.replay_memory.length() >= MEMORY_CAPACITY):
                    print('capacity exceeded')
                num_steps += 1
                ep_length += 1
                s = s_new
                ep_reward += reward
                x = self.experience_replay()
                q_sum += x
                if num_steps % TARGET_UPDATE == 0:
                    #print('update')
                    self.Q_target.load_state_dict(self.Q.state_dict())
            if EPSILON > 0.02:
                EPSILON *= EPSILON_DECAY
            
            ep_rewards.append(ep_reward)
            q_estimates.append((q_sum / ep_length))
            ep_rewards_temp.append(ep_reward)
            q_estimates_temp.append((q_sum / ep_length))
            if episode % 20 == 0:
                rewards = torch.as_tensor(ep_rewards_temp, dtype=torch.float)
                q = torch.as_tensor(q_estimates_temp, dtype=torch.float)
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
    env = gym.make('PongNoFrameskip-v4')
    env = wrappers.Create(env)
    Q = CNN((84, 84), 6).to(device)
    transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'terminal'])
    optimizer = torch.optim.Adam(Q.parameters(), lr=1e-4)
    dqn = DQN(env, Q, 2, optimizer)
    dqn.train()