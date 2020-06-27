import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distr
import gym
import matplotlib.pyplot as plt

def MLP(layer_sizes, act_fn=F.relu):
    layers = []
    for i in range(len(layer_sizes) - 1):
        if i == len(layer_sizes) - 1:
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        else:
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1], act_fn))
    return nn.Sequential(*layers)

def train(layer_sizes, env, lr=1e-2, epochs=10, batch_size=1000):
    env = gym.make(env)
    
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    layer_sizes.insert(0, obs_dim)
    layer_sizes.append(n_acts)
    net = MLP(layer_sizes)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def get_policy(obs):
        logits = net(obs)
        return distr.Categorical(logits=logits)

    def get_action(obs):
        policy = get_policy(obs)
        return policy.sample().item()

    def compute_loss(obs, actions, weights):
        log_prob = get_policy(obs).log_prob(actions)
        return -(log_prob * weights).mean()

    def train_one_epoch():
        batch_obs = []
        batch_rewards = []
        batch_actions = []
        first_batch_rendered = False

        # sample episodes
        for _ in range(0, batch_size):
            obs = env.reset()
            batch_obs.append(obs)
            total_ep_reward = 0.0
            step_rewards = []
            done = False

            # run an episode
            while not done:
                if not first_batch_rendered:
                    env.render()
                action = get_action(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, done, _ = env.step(action)
                if not done:
                    batch_obs.append(obs)
                batch_actions.append(action)
                step_rewards.append(reward)
                total_ep_reward += reward

            if not first_batch_rendered:
                first_batch_rendered = True
            
            # calculate reward to go
            for step_rew in step_rewards:
                batch_rewards.append(total_ep_reward - step_rew)

        # use average reward as baseline
        batch_weights = torch.as_tensor(batch_rewards, dtype=torch.float32)
        avg_reward = batch_weights.mean()
        batch_weights = batch_weights.sub_(avg_reward)

        #perform gradient step
        optimizer.zero_grad()
        loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                            actions=torch.as_tensor(batch_actions, dtype=torch.float32),
                            weights=batch_weights)
        loss.backward()
        optimizer.step()

        return avg_reward.item()

    avg_reward_list = []
    for _ in range(epochs):
        avg_reward = train_one_epoch()
        avg_reward_list.append(avg_reward)
        print(avg_reward)

    plt.plot(avg_reward_list)

    for _ in range(10):
        obs = env.reset()
        done = False
        while not done:
            env.render()
            action = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, reward, done, _ = env.step(action)

    env.close()
  
if __name__ == "__main__":
    train(layer_sizes=[10, 5], env='LunarLander-v2', epochs=50, batch_size=1000)
