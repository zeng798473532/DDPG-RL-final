import argparse
import os

import gym
import torch.cuda
import numpy as np
import time
from model import DDPG
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="MountainCarContinuous-v0",
                    help="Support Pendulum-v1, MountainCarContinuous-v0")
parser.add_argument("--render", type=bool, default=False, help="测试默认渲染")
parser.add_argument("--render-interval", type=int, default=2)
parser.add_argument("--pool-size", type=int, default=10000, help="ReplayPoolSize")
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--gamma", type=float, default=0.95, help="折扣率")
parser.add_argument("--actor-lr", type=float, default=0.001, help="actor-learning-rate")
parser.add_argument("--critic-lr", type=float, default=0.001, help="critic-learning-rate")
parser.add_argument("--tau", type=float, default=0.01, help="soft update网络所使用的权重参数")
parser.add_argument("--max-step", type=int, default=1000, help="Pendulum只允许走200步")
parser.add_argument("--warm-episode", type=int, default=2)
parser.add_argument("--episode", type=int, default=100)
parser.add_argument("--test-episode", type=int, default=10)
parser.add_argument("--load-path", type=str, default="", help="like: models/ddpg, ignore:_actor.pth or _critic.pth")
parser.add_argument("--noise", type=str, default="Gauss", help="Gauss or OU or None")
parser.add_argument("--eps", type=float, default=0.9)
parser.add_argument("--eps-decay", type=float, default=0.95)
parser.add_argument("--eps-min", type=float, default=0.05, help="最小噪声影响权重")
parser.add_argument("--save", action='store_true', default=True)
parser.add_argument("--train", action='store_true', default=False, help="load model默认不训练")
parser.add_argument("--eval", action='store_true', default=False, help="load model默认测试")
parser.add_argument("--cuda", action='store_true', default=False)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
# device = torch.device("cpu")
print("device:", device)
now = time.localtime(time.time())
now = f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}"


def which_to_render(a):
    return True


env = gym.make(args.env)
# env = gym.wrappers.Monitor(
#     env,
#     f'./wrapper/Training-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}',
#     force=True,
#     video_callable=which_to_render
# )
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]
print(f"Env: {args.env}")
print(f"state_dim:{state_dim}\naction_dim:{action_dim}\nmax_action:{max_action}\nmin_action:{min_action}")

cut = 1
seed = 1024
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
total_rewards = []
test_rewards = []

agent = DDPG(state_dim, action_dim, device, env, args)
if args.load_path != "":
    agent.load_model(args.load_path)

if args.load_path == "" or (args.load_path != "" and args.train):
    if args.warm_episode > 0:
        print(f"Warm-up for {args.warm_episode} episodes...")
    warmenv = gym.make(args.env)
    for episode in range(args.warm_episode):
        state = warmenv.reset()
        for step in range(args.max_step):
            # if args.render:
                # warmenv.render()
            action = agent.random_action()
            next_state, reward, done, _ = warmenv.step(action)
            agent.save_replay(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    if args.warm_episode > 0:
        print("Warm-up Done...")

loss_mse = []
loss_J = []
if args.load_path == "" or (args.load_path != "" and args.train):
    print("Begin Learning..")
    begin = time.time()
    for episode in range(args.episode):
        start = time.time()
        total_reward = 0
        state = env.reset()
        lossLs = []
        lossJs = []
        for step in range(args.max_step):
            if args.render and (episode + 1) % args.render_interval == 0:
                env.render()
            action = agent.get_action(state, noise=True)
            next_state, reward, done, _ = env.step(action)
            agent.save_replay(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            lossL, lossJ = agent.learn()
            lossLs.append(lossL) if lossL is not None else None
            lossJs.append(lossJ) if lossL is not None else None
            if done:
                break
        end = time.time()
        agent.update_eps()  # 每个episode更新一次噪声权重，从0.95递减到0.05约58次
        total_rewards.append(total_reward)
        loss_mse.append(sum(lossLs)/len(lossLs))
        loss_J.append(sum(lossJs)/len(lossJs))
        print(f"Episode {episode} rewards:{total_reward:.2f}, spend:{(end - start):.2f}seconds.",
              f"Critic MSEloss:{loss_mse[-1]:.2f},Actor Meanloss:{loss_J[-1]:.2f}")
        # time.sleep(1)
    print("total_rewards:", total_rewards, "Spend: ", (time.time() - begin) / 60, "minutes")

    plt.figure()
    plt.plot([i*cut for i in range(args.episode//cut)],
             [np.mean(total_rewards[i*cut:(i+1)*cut]) for i in range(args.episode//cut)], color='red')
    plt.title(f"Rewards-Training-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}")
    plt.savefig(f"Rewards-Training-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}.png")
    plt.show()

if args.save and (args.load_path == "" or (args.load_path != "" and args.train)):
    agent.save_model()

if args.test_episode > 0 or args.eval or args.load_path:
    print(f"Test for {args.test_episode} episodes...Please confirm. ")
    # os.system("pause")
    testenv = gym.make(args.env)
    testenv = gym.wrappers.Monitor(
        testenv,
        f'./wrapper/Testing-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}',
        force=True,
        video_callable=which_to_render,
        mode="evaluation"
    )
    for episode in range(args.test_episode):
        state = testenv.reset()
        test_reward = 0
        for step in range(args.max_step):
            testenv.render()
            action = agent.get_action(state, noise=False)
            next_state, reward, done, _ = testenv.step(action)
            agent.save_replay(state, action, reward, next_state, done)
            test_reward += reward
            state = next_state
            # time.sleep(0.05)
            if done:
                break
        test_rewards.append(test_reward)
        print(f"Test {episode} rewards:", "%.2f" % test_reward)
        time.sleep(1)

    print("Test Done...")
    plt.figure()
    plt.plot([i*cut for i in range(args.test_episode//cut)],
             [np.mean(test_rewards[i*cut:(i+1)*cut]) for i in range(args.test_episode//cut)], color='red')
    # plt.legend(loc='best')
    plt.title(f"Rewards-Testing-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}")
    plt.savefig(f"Rewards-Testing-{args.env}-{args.noise}-Gamma{args.gamma}-Tau{args.tau}-{now}.png")
    plt.show()
