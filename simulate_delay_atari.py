from environments.atari_env import *

import numpy as np
import time
import sys
import torch
from model.agents import *
import argparse
import pickle
import copy
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

parser = argparse.ArgumentParser(description='Experiments Delay in Tetris')
### model details
parser.add_argument('--model', type=str, default="dqn", help='random or cnn')
parser.add_argument('--arch', type=str, default="impala", help='architecture: nature or impala')
parser.add_argument('--k', type=float, default=1.0, help='number of filters per layer normalized by standard impala, nature default is 2.0')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--opt', type=str, default="Adam", help='which optimizer would you like to use? Adam or SGD')
parser.add_argument('--cuda', type=str, default="no", help='gpu on?')
parser.add_argument('--explore', type=int, default=100000, help='exploration phase length')
parser.add_argument('--target_freq', type=int, default=10000, help='target update frequency (number of updates)')
parser.add_argument('--reservoir', type=str, default="no", help='yes = reservoir sampling, no = recency sampling')
parser.add_argument('--n', type=int, default=3, help='n for n-step learning algorithms')

### environment details
parser.add_argument('--fps', type=float, default=0.0, help='frames per second: 0.0 for now waiting on hardware or 59.7275 for gameboy simulation')

### tetris environment details
parser.add_argument('--game', type=str, default="PongNoFrameskip-v4", help='env name')
parser.add_argument('--pretrain_steps', type=int, default=100, help='number of pretraining steps')
parser.add_argument('--delay', type=int, default=0, help='number of pretraining steps')


parser.add_argument('--noops', type=int, default=0, help='noop actions per actual action')
parser.add_argument('--num_actions', type=int, default=6, help='number of actions (default is 6)')

parser.add_argument('--default_action', type=int, default=0, help='the noop action is 0 for these games')


### resource constraint details 
parser.add_argument('--obs_history', type=int, default=1, help='number of observations used in history')
parser.add_argument('--memory', type=int, default=1000000, help='memory size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--updates', type=int, default=1, help='updates per step in environment')


args = parser.parse_args()

args.cuda = True if args.cuda == 'yes' else False
args.reservoir = True if args.reservoir == 'yes' else False
    
print("args:", args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

seed = args.seed
avg_rewards = []
avg_lines = []
obs_buffer = []
action_buffer = []
reward_buffer = []
next_obs_buffer = []

for run in range(1):
    args.seed = seed + run
    if args.model == "random":
        policy = RandomAgent(args)
    if args.model == "noop":
        policy = NoopAgent(args)
    elif args.model == "dqn":
        policy = DQNAgent(args, obs_buffer=obs_buffer, action_buffer=action_buffer, reward_buffer=reward_buffer,
                          next_obs_buffer=next_obs_buffer)
        print("parameter count:", count_parameters(policy.network))
    elif args.model == "rainbow":
        policy = RainbowAgent(args, obs_buffer=obs_buffer, action_buffer=action_buffer, reward_buffer=reward_buffer,
                          next_obs_buffer=next_obs_buffer)
        print("parameter count:", count_parameters(policy.dqn))
    else:
        print("model not recognized!", args.model)




    game_id = args.game
    env = gym.make("Atari-v0", state=game_id, render=False, noops=args.noops, obs_history=args.obs_history, fps=args.fps)



    wins = 0
    losses = 0
    steps = 0
    episodes = 0
    positive_rewards = 0
    negative_rewards = 0
    begin_time = time.time()
    ep_reward = []
    for episode in range(2000):
        obs = env.reset()
        zero_pad = np.zeros(shape=obs.shape)
        obs_history = []
        for past in range(args.delay):
            obs_history.append(zero_pad)

        obs_history.append(obs)
        if args.delay > 0:
            obs_history = obs_history[1:]

        done = False
        reward = 0.0
        ep_steps = 0
        while not done:
            action = policy.act(obs_history[0])
            steps += 1
            ep_steps += 1
            next_obs, rew, done, info = env.step(action)
            reward += rew
            policy.add_items(obs_history[0], action, rew, next_obs, done)
            policy.num_env_steps += 1
            if steps % 4 == 0: # it is typical to update the model every 4 steps in Atari
                policy.learn(obs_history[0], action, rew, next_obs)

            obs = copy.deepcopy(next_obs)
            obs_history.append(obs)
            obs_history = obs_history[1:]

        time_elapsed = time.time() - begin_time
        ep_reward.append(reward)
        print("run=",run+1,"episodes=", episode + 1, "ep_reward=", reward, "total_steps=", steps,
              "time_elapsed=", time_elapsed)

        print("epsilon=", policy.epsilon, "greedy_actions=", policy.greedy_actions)
        env.close()
        env = gym.make("Atari-v0", state=game_id, render=False, noops=args.noops, obs_history=args.obs_history,
                       fps=args.fps)
        sys.stdout.flush()

    env.close()
    avg_reward = np.array(ep_reward).mean()
    avg_rewards.append(avg_reward)


print("Results:","Average_Episodic_Reward=",np.array(avg_rewards).mean(),"STD=",np.array(avg_rewards).std())
