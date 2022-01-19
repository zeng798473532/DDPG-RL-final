import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from Replay_Memory import Replay_Memory


class Actor(nn.Module):
    """
    Actor: 输入当前的状态s，输出当前应该执行的行动action
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, action_dim)

    def forward(self, state):
        y = F.relu(self.linear1(state))
        y = F.relu(self.linear2(y))
        y = torch.tanh(self.linear3(y))
        return y

class Critic(nn.Module):
    """
    Critic: 输入当前的状态以及经过Actor的判断的行动action，输出执行该action的价值value
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class DDPG(object):
    def __init__(self, state_dim, action_dim, device, env, args):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)  # 目标网络，缓慢更新权重
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)  # 目标网络，缓慢更新权重
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.replay_memory = Replay_Memory(args.pool_size, args.batch_size)
        self.args = args
        self.device = device
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.env = env
        self.noise = None
        self.eps = args.eps
        if args.noise == "OU":
            self.noise = OUNoise(action_dim)
        elif args.noise == "Gauss":
            self.noise = GaussNoise(action_dim)
        

    def get_action(self, state, noise=True):
        """
        根据状态获得当前所采取的action，同时加入随机值进行探索，注意需要限制范围
        :param noise: 是否加入噪声
        :param state:
        :return:action
        """
        state = torch.tensor(state).to(self.device)
        action = self.actor(state)
        action = action.cpu().detach().numpy()
        if self.args.noise != "" and self.args.noise != "None" and noise:
            action += self.noise.noise() * self.eps
            action.clip(self.env.action_space.low[0], self.env.action_space.high[0])
        return action

    def random_action(self):
        """
        采取随机值作为action，用于热身步骤
        :return: action
        """
        max_action = self.env.action_space.high[0]
        min_action = self.env.action_space.low[0]
        t = np.random.uniform(min_action, max_action)
        return np.array(t).reshape((1,))

    def learn(self):
        # 先采样若batchsize个samples = random.sample(self.buffer, self.batch_size)
        # 梯度下降更新参数
        # soft update目标网络参数
        if len(self.replay_memory.pool) < self.replay_memory.batch_size:
            return None, None
        rand_data = self.replay_memory.sample()
        states, actions, rewards, next_states, done = zip(*rand_data)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)  # 调用网络需要使用tensor
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).view(self.args.batch_size, -1).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).view(self.args.batch_size, -1).to(self.device)
        a1 = self.actor_target(next_states)
        y_target = rewards + self.args.gamma * self.critic_target(next_states, a1).detach()
        y_now = self.critic(states, actions)
        # detach到底有什么用？两个y都有detach就不能跑了
        loss_fun = nn.MSELoss()
        lossL = loss_fun(y_now, y_target)
        self.critic_optim.zero_grad()
        lossL.backward()
        self.critic_optim.step()

        lossJ = -self.critic(states, self.actor(states)).mean()
        self.actor_optim.zero_grad()
        lossJ.backward()
        self.actor_optim.step()
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        return lossL, lossJ
       
    def update_eps(self):
        self.eps = max(self.args.eps_min, self.eps * self.args.eps_decay)

    def save_replay(self, state, action, reward, next_state, done):
        self.replay_memory.append(state, action, reward, next_state, done)
        pass

    def soft_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.args.tau) + param.data * self.args.tau)

    def hard_update(self, net_target, net):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, prefix=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        now = time.localtime(time.time())
        name = f"models/{self.args.env}_{self.args.noise}_{now.tm_year}-{now.tm_mon}-{now.tm_mday}"
        path_actor = f"{name}_actor.pth"
        path_critic = f"{name}_critic.pth"
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)

    def load_model(self, prefix):
        path_actor = f"{prefix}_actor.pth"
        path_critic = f"{prefix}_critic.pth"
        self.actor.load_state_dict(torch.load(path_actor))
        self.critic.load_state_dict(torch.load(path_critic))
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

# reference: https://github.com/cannoliWang/DDPG
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class GaussNoise:
    def __init__(self, action_dimension, mu=0, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        tmp = np.random.normal(self.mu, self.sigma, size=self.action_dimension)
        return tmp
