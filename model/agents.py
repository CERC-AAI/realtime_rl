import random
from model.architectures import *
from torch.autograd import Variable
import torch.nn.functional as F



import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from model.segment_tree import MinSegmentTree, SumSegmentTree

class RandomAgent:

    def __init__(self, args):
        self.num_actions = args.num_actions
        random.seed(args.seed)
        self.epsilon = 1.0
        self.greedy_actions = 0
        self.num_env_steps = 0

    def act(self,obs):
        action = random.randint(0,self.num_actions-1)
        return action

    def learn(self,obs,action,reward,next_obs):
        null = 0

    def add_items(self, obs, action, reward, next_obs,done=None):
        null = 0

class NoopAgent:

    def __init__(self, args):
        self.num_actions = args.num_actions
        random.seed(args.seed)
        self.action = args.default_action
        self.epsilon = 0.0
        self.greedy_actions = 0
        self.num_env_steps = 0

    def act(self,obs):
        self.greedy_actions += 1
        return self.action

    def learn(self,obs,action,reward,next_obs):
        null = 0

    def add_items(self, obs, action, reward, next_obs,done=None):
        null = 0





class DQNAgent:

    def __init__(self, args, obs_buffer=[], action_buffer=[], reward_buffer=[], next_obs_buffer=[]):
        self.num_actions = args.num_actions
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        self.gamma = args.gamma
        self.batchSize = args.batch_size
        self.lr = args.lr
        self.memory = args.memory
        self.updates = args.updates
        self.cuda = args.cuda

        if args.arch == "nature":
            self.network = NatureCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
            self.target = NatureCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        elif args.arch == "impala":
            self.network = ImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
            self.target = ImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        elif args.arch == "deepimpala":
            self.network = DeepImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
            self.target = DeepImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        else:
            print("architecture not recognized!!!", args.arch)

        if self.cuda:
            self.network = self.network.cuda()
            self.target = self.target.cuda()

        self.target.load_state_dict(self.network.state_dict())
        if args.opt == "Adam":
            self.opt = torch.optim.Adam(self.network.parameters(), lr=args.lr)
        elif args.opt == "SGD":
            self.opt = torch.optim.SGD(self.network.parameters(), lr=args.lr)
        else:
            print(args.opt, "Not recognized!")

        self.num_update_steps = 0
        self.num_env_steps = 0
        self.greedy_actions = 0

        self.action_buffer = action_buffer

        self.reward_buffer = reward_buffer

        self.target_freq = args.target_freq

        self.obs_buffer = obs_buffer

        self.next_obs_buffer = next_obs_buffer

        self.explore = args.explore
        self.eps_init = 1.0
        self.eps_end = 0.05
        self.reservoir = args.reservoir

    def getEpsilon(self):
        current_item = self.num_env_steps
        eps = self.eps_init - (self.eps_init - self.eps_end) * min(current_item / self.explore, 1)
        return round(eps, 2)

    def act(self, obs):
        with torch.no_grad():
            obs_var = Variable(torch.from_numpy(np.array(obs))).float().unsqueeze(0).transpose(1, 3).transpose(2, 3)
            if self.cuda:
                obs_var = obs_var.cuda()

            self.epsilon = self.getEpsilon()
            if random.random() > self.epsilon:
                if self.cuda:
                    qs = self.network.forward(obs_var)[0].cpu().data.numpy()
                else:
                    qs = self.network.forward(obs_var)[0].data.numpy()
                # print("qs=",qs)
                action = qs.argmax()
                self.greedy_actions += 1
            else:
                action = random.randint(0, self.num_actions - 1)
            return action

    def add_items(self, obs, action, reward, next_obs, done=None):
        obs_var = Variable(torch.from_numpy(np.array(obs))).float().unsqueeze(0).transpose(1, 3).transpose(2, 3)
        next_obs_var = Variable(torch.from_numpy(np.array(next_obs))).float().unsqueeze(0).transpose(1, 3).transpose(2,
                                                                                                                     3)
        my_obs = obs_var.data.numpy()
        my_next_obs = next_obs_var.data.numpy()
        if self.num_env_steps >= self.memory:
            if self.reservoir:
                myitem = random.randint(0, self.num_env_steps)
                if myitem < self.memory:
                    self.obs_buffer[myitem] = my_obs
                    self.next_obs_buffer[myitem] = my_next_obs
                    self.action_buffer[myitem] = action
                    self.reward_buffer[myitem] = reward
            else:
                myitem = self.num_env_steps % self.memory
                self.obs_buffer[myitem] = my_obs
                self.next_obs_buffer[myitem] = my_next_obs
                self.action_buffer[myitem] = action
                self.reward_buffer[myitem] = reward

        else:
            self.obs_buffer.append(my_obs)
            self.next_obs_buffer.append(my_next_obs)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)

    def getItems(self, indexes):
        actions = []
        rewards = []
        observations = []
        next_observations = []
        for ind in indexes:
            actions.append(self.action_buffer[ind])
            rewards.append(self.reward_buffer[ind])
            observations.append(self.obs_buffer[ind])
            next_observations.append(self.next_obs_buffer[ind])

        np_a = np.array(actions).astype(int)
        np_reward = np.array(rewards).astype(float)
        np_s = np.concatenate(observations, axis=0).astype(float)
        np_ns = np.concatenate(next_observations, axis=0).astype(float)
        s = Variable(torch.from_numpy(np_s)).float()
        a = torch.from_numpy(np_a).long().view(-1, 1)
        r = torch.from_numpy(np_reward).float().view(-1, 1)
        ns = Variable(torch.from_numpy(np_ns)).float()
        if self.cuda:
            s = s.cuda()
            a = a.cuda()
            r = r.cuda()
            ns = ns.cuda()

        return s, a, r, ns

    def update_network(self):
        indexes = np.random.randint(min(self.num_env_steps, self.memory), size=self.batchSize)
        s, a, r, ns = self.getItems(indexes)
        # print("sizes=",s.size(),a.size(),r.size(),ns.size())

        self.network.zero_grad()
        qs = self.network.forward(s)
        q = qs.gather(dim=1, index=a)
        with torch.no_grad():
            target_qs = self.target.forward(ns)
            target_q = self.gamma * target_qs.max(1).values.view(-1, 1) + r

        loss = F.smooth_l1_loss(q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10.0)
        self.opt.step()

    def learn(self, obs, action, reward, next_obs):
        #self.add_items(obs, action, reward, next_obs)
        #self.num_env_steps += 1
        for i in range(self.updates):
            self.update_network()
            self.num_update_steps += 1
            if self.num_update_steps % self.target_freq == 0:
                self.target.load_state_dict(self.network.state_dict())




class ReplayBuffer:
    """A simple numpy replay buffer.
    Parameters
    ---------
    obs_dim: list[int]
        Observation shape
    size: int
        # maximum number of elements in buffer
    batch_size: int
        batch_size
    n_step: int
        number of step used for N-step learning
    gamma: float
        gamma value
    """

    def __init__(
            self,
            obs_dim: List[int],
            size: int = 1024,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool] or None:
        """Store a new experience in the buffer"""
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return

        # make a n-step transition
        # take the n-reward, n-observation and n-done
        rew, next_obs, done = self._get_n_step()
        # take the 1-observation and 1-action
        obs, act = self._get_first_step()

        # store the transition
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return transition

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Sample a batch from the buffer"""
        assert len(self) >= self.batch_size
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,  # # need this for priority updating
        )

    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        """Sample a batch given some fixed idxs"""
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def _get_n_step(self) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = self.n_step_buffer[-1][-3:]

        for transition in reversed(list(self.n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]
            # update the reward
            rew = r + self.gamma * rew * (1 - d)
            # if done == 1: next_obs is the first observation where done == 1
            # if done == 0: next_obs is the n-observation
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def _get_first_step(self) -> Tuple[np.int64, np.ndarray]:
        """Return first step obs and act."""
        # info of the first transition
        obs, act = self.n_step_buffer[0][:2]

        return obs, act

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    Attributes:
        max_priority: float
            max priority
        tree_ptr: int
            next index of tree
        alpha: float
            alpha parameter for prioritized replay buffer
        sum_tree: SumSegmentTree
            sum tree for prior
        min_tree: MinSegmentTree
            min tree for min prior to get max weight
    """

    def __init__(
            self,
            obs_dim: List[int],
            size: int = 1024,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority = 1.0
        self.tree_ptr = 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """Store an experience and its priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        # samples transitions indices
        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        # importance sampling weights
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,  # need this for priority updating
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self))
        segment = p_total / self.batch_size

        # perform a random sample in each segment
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upper_bound = random.uniform(a, b)
            idx = self.sum_tree.find_prefixsum_idx(upper_bound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.



    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class RainbowAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """
    def __init__(self, args, obs_buffer=[], action_buffer=[], reward_buffer=[], next_obs_buffer=[]):
        self.num_actions = args.num_actions
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # PER parameters
        alpha = 0.2
        beta = 0.6
        prior_eps = 1e-6
        # Categorical DQN parameters
        v_min = 0.0
        v_max = 200.0
        atom_size = 51
        # N-step Learning
        n_step = 3

        self.gamma = args.gamma
        self.lr = args.lr
        self.cuda = args.cuda

        self.dqn = RainbowImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)
        self.dqn_target = RainbowImpalaCNN(actions=self.num_actions, in_channels=args.obs_history, k=args.k)

        self.dqn_target.load_state_dict(self.dqn.state_dict())

        if self.cuda:
            self.dqn = self.dqn.cuda()
            self.dqn_target = self.dqn_target.cuda()


        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=args.lr)

        self.num_update_steps = 0
        self.num_env_steps = 0
        self.greedy_actions = 0

        self.action_buffer = action_buffer

        self.reward_buffer = reward_buffer

        self.target_freq = args.target_freq
        self.target_update = args.target_freq
        self.updates = args.updates

        self.obs_buffer = obs_buffer

        self.next_obs_buffer = next_obs_buffer

        self.explore = args.explore
        self.eps_init = 1.0
        self.eps_end = 0.05
        self.reservoir = args.reservoir

        self.batch_size = args.batch_size
        #self.target_update =
        self.seed = args.seed
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        obs_dim = [1, 84, 84]
        self.memory = PrioritizedReplayBuffer(
            obs_dim, args.memory, self.batch_size, alpha=alpha, gamma=self.gamma
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, args.memory, self.batch_size, n_step=n_step, gamma=self.gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        print("state=",state.shape)
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        print("action=",selected_action)
        selected_action = selected_action.detach().cpu().numpy()

        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def add_items(self, raw_obs, act, rew, raw_next_obs, done):
        obs =  np.transpose(raw_obs, (2, 0, 1))
        next_obs = np.transpose(raw_next_obs, (2, 0, 1))
        self.transition = [obs, act, rew, next_obs, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
        # 1-step transition
        else:
            one_step_transition = self.transition

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(frame_idx / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)
                scores.append(score)
                score = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)

        self.env.close()

    def learn(self, obs, action, reward, next_obs):
        #self.add_items(obs, action, reward, next_obs)
        #self.num_env_steps += 1
        for i in range(self.updates):
            self.num_update_steps += 1
            if self.num_update_steps >= self.batch_size:
                loss = self.update_model()
                if self.num_update_steps % self.target_freq == 0:
                    self._target_hard_update()

    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        naive_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        state, _ = self.env.reset(seed=self.seed)
        done = False
        score = 0

        while not done:
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

        # reset
        self.env = naive_env

    def act(self, obs): # added
        with torch.no_grad():
            obs_var = Variable(torch.from_numpy(np.array(obs))).float().unsqueeze(0).transpose(1, 3).transpose(2, 3)
            if self.cuda:
                obs_var = obs_var.cuda()

            self.epsilon = self.getEpsilon()
            if random.random() > self.epsilon:
                if self.cuda:
                    qs = self.dqn.forward(obs_var)[0].cpu().data.numpy()
                else:
                    qs = self.dqn.forward(obs_var)[0].data.numpy()
                # print("qs=",qs)
                action = qs.argmax()
                self.greedy_actions += 1
            else:
                action = random.randint(0, self.num_actions - 1)
            return action

    def getEpsilon(self): #added
        current_item = self.num_env_steps
        eps = self.eps_init - (self.eps_init - self.eps_end) * min(current_item / self.explore, 1)
        return round(eps, 2)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            #print("ns=",next_state.size())
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
