import random
import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d, tanh
from random import shuffle
import sys
from model.architectures import *
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time
from torch.multiprocessing import RawValue
import torch.optim as optim
from collections import defaultdict
import pdb
import threading
import cv2 as cv
import torch
import gym
import os
from collections import deque
import random
import torch.multiprocessing as mp

import math

import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class Net(torch.nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.gamma = 0.99
        self.batchSize = args.batch_size
        self.actions = args.num_actions
        self.lr = args.lr
        self.memory = args.memory
        self.age = mp.Manager().Value('i', 0)
        self.num_actions = args.num_actions
        if args.arch == "nature":
            self.network = NatureCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
            self.target = NatureCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        elif args.arch == "impala":
            self.network = ImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
            self.target = ImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        else:
            print("architecture not recognized!!!", args.arch)

        self.target.load_state_dict(self.network.state_dict())
        self.memory = args.memory
        self.state_size = args.state_size
        self.item = mp.Manager().Value('i', 0)
        self.reservoir = args.reservoir

        print("state_size:", self.state_size[2], self.state_size[0], self.state_size[1])
        if args.opt == "Adam":
            self.opt = torch.optim.Adam(self.network.parameters(), lr=args.lr)
        elif args.opt == "SGD":
            self.opt = torch.optim.SGD(self.network.parameters(), lr=args.lr)
        elif args.opt == "SharedAdam":
            self.opt = SharedAdam(self.network.parameters(), lr=args.lr)
        else:
            print(args.opt, "Not recognized!")
        self.network.share_memory()
        self.target.share_memory()

        self.target_freq = args.target_freq
        self.updates_freq = 0
        self.explore = args.explore
        self.eps_init = args.eps_init
        self.eps_end = args.eps_end
        self.greedy_actions = mp.Manager().Value('i', 0)

    def getEpsilon(self, env_step):
        eps = self.eps_init - (self.eps_init - self.eps_end) * min(env_step / self.explore, 1)
        return round(eps, 2)

    # def getAction(self, s, env_step):
    #     with torch.no_grad():
    #         self.epsilon = self.getEpsilon(env_step)
    #         if random.random() > self.epsilon:
    #             qs = self.network.forward(s)
    #             action = qs.argmax()
    #             # print("self.greedy_actions.value",self.greedy_actions.value)
    #             self.greedy_actions.value += 1
    #         else:
    #             # print("inside random action")
    #             action = random.randint(0, self.actions - 1)
    #         return action


    def getAction(self, s, env_step, greedy_inf_mean_time, num_greedy_act):
        with torch.no_grad():
            self.epsilon = self.getEpsilon(env_step)
            # print("self epsilon:", self.epsilon)

            # num_greedy_act += 1
            # before = time.time()
            # qs = self.network.forward(s)
            # action = qs.argmax()
            # after = time.time()
            # greedy_inf_mean_time = (greedy_inf_mean_time*(num_greedy_act-1) + (after - before))/num_greedy_act
            # # print("greedy inf mean time:", greedy_inf_mean_time)
            # self.greedy_actions.value += 1
            if np.random.rand() >= self.epsilon:
                num_greedy_act += 1
                before = time.time()
                qs = self.network.forward(s)
                action = qs.argmax()
                after = time.time()
                greedy_inf_mean_time = (greedy_inf_mean_time*(num_greedy_act-1) + (after - before))/num_greedy_act
                # print("greedy inf mean time:", greedy_inf_mean_time)
                self.greedy_actions.value += 1
            else:
                # print("inside random action")
                action = random.randint(0, self.actions - 1)
                # print("greedy inf mean time:", greedy_inf_mean_time)
                # if greedy_inf_mean_time > 0.0:
                #     time.sleep(greedy_inf_mean_time)
            # print("got action done!!!")
            return action, greedy_inf_mean_time, num_greedy_act

    def learn(self, s, a, r, ns, learn_id, lock, device):
        # get random batch of transitions
        # s, a, r, ns, _ = output_queue.get()
        # run one step of optimization on batch of transitions
        # print("Inside learn function!")
        if self.args.arch == "nature":
            local_network = NatureCNN(actions=self.num_actions, in_channels=self.args.obs_history, k=self.args.k).to(
                device)
            local_target = NatureCNN(actions=self.num_actions, in_channels=self.args.obs_history, k=self.args.k).to(
                device)
        elif self.args.arch == "impala":
            local_network = ImpalaCNN(actions=self.num_actions, in_channels=self.args.obs_history, k=self.args.k).to(
                device)
            local_target = ImpalaCNN(actions=self.num_actions, in_channels=self.args.obs_history, k=self.args.k).to(
                device)
        else:
            print("architecture not recognized!!!", self.args.arch)
        local_process_count = 0
        local_network.load_state_dict(self.network.state_dict())
        local_target.load_state_dict(self.target.state_dict())
        if self.args.num_learn == 1:

            self.age.value += 1
            local_network.zero_grad()
            qs = self.network.forward(s)
            q = qs.gather(dim=1, index=a.long().view(-1, 1))
            with torch.no_grad():
                target_qs = self.target.forward(ns)
                target_q = self.gamma * target_qs.max(1).values.view(-1, 1) + r.view(-1, 1)
            loss = F.smooth_l1_loss(q, target_q)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_network.parameters(), 10.0)

            for local_net_param, shared_net_param, local_tar_param, shared_tar_param in zip(local_network.parameters(),
                                                                                            self.network.parameters(),
                                                                                            local_target.parameters(),
                                                                                            self.target.parameters()):
                if shared_net_param.grad is None:
                    shared_net_param._grad = local_net_param.grad
                if shared_tar_param.grad is None:
                    shared_tar_param._grad = local_tar_param.grad

            self.opt.step()
        else:
            # lock.acquire()

            # print("learn ID at the start:", learn_id)
            # print("In multi process learning!!!")
            # print("action:", a)
            self.age.value += 1
            local_network.zero_grad()
            qs = self.network.forward(s)
            q = qs.gather(dim=1, index=a.long().view(-1, 1))
            with torch.no_grad():
                target_qs = self.target.forward(ns)
                target_q = self.gamma * target_qs.max(1).values.view(-1, 1) + r.view(-1, 1)
            loss = F.smooth_l1_loss(q, target_q)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_network.parameters(), 10.0)

            for local_net_param, shared_net_param, local_tar_param, shared_tar_param in zip(local_network.parameters(),
                                                                                            self.network.parameters(),
                                                                                            local_target.parameters(),
                                                                                            self.target.parameters()):
                if shared_net_param.grad is None:
                    shared_net_param._grad = local_net_param.grad
                if shared_tar_param.grad is None:
                    shared_tar_param._grad = local_tar_param.grad
            lock.acquire()
            self.opt.step()
            # print("learn ID at the end:", learn_id)
            lock.release()

        self.updates_freq += 1
        # update target network
        if self.age.value % self.target_freq == 0:
            self.target.load_state_dict(self.network.state_dict())

