from environments.tetris_env import *

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
parser.add_argument('--explore', type=int, default=16001, help='exploration phase length')
parser.add_argument('--target_freq', type=int, default=10000, help='target update frequency (number of updates)')
parser.add_argument('--reservoir', type=str, default="no", help='yes = reservoir sampling, no = recency sampling')
parser.add_argument('--n', type=int, default=3, help='n for n-step learning algorithms')

### environment details
parser.add_argument('--fps', type=float, default=0.0, help='frames per second: 0.0 for now waiting on hardware or 59.7275 for gameboy simulation')

### tetris environment details
parser.add_argument('--level', type=int, default=0, help='level number')
parser.add_argument('--pretrain_steps', type=int, default=16000, help='number of pretraining steps')
parser.add_argument('--delay', type=int, default=0, help='number of pretraining steps')


parser.add_argument('--noops', type=int, default=0, help='noop actions per actual action')
parser.add_argument('--num_actions', type=int, default=6, help='number of actions (default is 6)')

parser.add_argument('--default_action', type=int, default=6, help='the noop action is 6 for this game')


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

orig_obs_buffer = []
orig_action_buffer = []
orig_reward_buffer = []
orig_next_obs_buffer = []

noops_buffer = []
if args.pretrain_steps > 0:
    for episode in range(1):

        with open('tetris_episode' + str(episode) + '.pkl', 'rb') as f:
            data = pickle.load(f)

        tmp_action = -1
        num_noops = 0
        reward_sum = 0.0
        for obs, action, reward, next_obs in data:
            if action == args.default_action:
                num_noops += 1
                tmp_next_obs = copy.deepcopy(next_obs)
                reward_sum += reward
            else:
                if tmp_action != -1:
                    #if num_noops == args.noops:
                    orig_obs_buffer.append(tmp_obs)
                    orig_action_buffer.append(tmp_action)
                    orig_reward_buffer.append(reward_sum)
                    orig_next_obs_buffer.append(tmp_next_obs)
                    noops_buffer.append(num_noops)

                tmp_action = action
                tmp_obs = copy.deepcopy(obs)
                reward_sum = reward
                tmp_next_obs = copy.deepcopy(next_obs)
                num_noops = 0

        #if num_noops == args.noops and num_noops > 0:
        orig_obs_buffer.append(tmp_obs)
        orig_action_buffer.append(tmp_action)
        orig_reward_buffer.append(reward_sum)
        orig_next_obs_buffer.append(tmp_next_obs)
        noops_buffer.append(num_noops)
        sys.stdout.flush()

        obs_buffer = []
        action_buffer = []
        reward_buffer = []
        next_obs_buffer = []
        zero_pad = np.zeros(shape=tmp_obs.shape)
        for ind in range(len(orig_obs_buffer)):
            obs_ind = ind - args.delay
            if obs_ind >= 0:
                obs = orig_obs_buffer[obs_ind]
            else:
                obs = copy.deepcopy(zero_pad)
                print("padding with zeros:",obs_ind, ind)

            act = orig_action_buffer[ind]
            rew = orig_reward_buffer[ind]
            nobs = orig_next_obs_buffer[ind]
            obs_buffer.append(obs)
            action_buffer.append(act)
            reward_buffer.append(rew)
            next_obs_buffer.append(nobs)


    print("data loaded:", len(obs_buffer), len(action_buffer), len(reward_buffer), len(next_obs_buffer))
    print("noops:", np.array(noops_buffer).mean(),"dev:",np.array(noops_buffer).std())



seed = args.seed
avg_rewards = []
avg_lines = []
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
    else:
        print("model not recognized!", args.model)




    if ("dqn" in args.model or "bc" in args.model) and args.pretrain_steps > 0:
        before_time = time.time()
        policy.num_env_steps = len(obs_buffer)
        policy.explore = 1

        for i in range(args.pretrain_steps):
            policy.update_network()

        after_time = time.time()
        print(args.pretrain_steps, "learning steps completed in",after_time-before_time,"seconds")
        policy.num_env_steps = len(obs_buffer)
        policy.explore = args.explore - args.pretrain_steps

    level_id = "TypeA_Level" + str(args.level)
    env = gym.make("Tetris-v0", state=level_id, render=False, noops=args.noops, obs_history=args.obs_history, fps=args.fps)



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
        # print(len(obs_history))

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
            policy.learn(obs_history[0], action, rew, next_obs)
            obs = copy.deepcopy(next_obs)
            obs_history.append(obs)
            obs_history = obs_history[1:]

        level = info['level']
        time_elapsed = time.time() - begin_time
        ep_reward.append(reward)
        print("run=",run+1,"episodes=", episode + 1, "ep_reward=", reward, "ep_level=", level, "total_steps=", steps,
              "time_elapsed=", time_elapsed)
        print("epsilon=", policy.epsilon, "greedy_actions=", policy.greedy_actions)
        env.close()
        env = gym.make("Tetris-v0", state=level_id, render=False, noops=args.noops, obs_history=args.obs_history,
                       fps=args.fps)
        sys.stdout.flush()

    env.close()
    avg_reward = np.array(ep_reward).mean()
    avg_rewards.append(avg_reward)

print("Results:","Average_Episodic_Reward=",np.array(avg_rewards).mean(),"STD=",np.array(avg_rewards).std())
