import os
import time
import json
from environments.pokemon_gen1_env import *
from environments.tetris_env import *
import numpy as np
import random
import time
import sys
import torch
import pickle
import argparse
import torch.multiprocessing as mp
import functools
from torch.autograd import Variable
import gym
import retro
import importlib
import pdb
import concurrent
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

class Experiment:
    def __init__(self, args, model, greed_act):

        self.model = model
        self.args = args
        self.lock = mp.Manager().Lock()
        self.buffer_idx = mp.Manager().Value('i', 0)
        self.action_default = mp.Manager().Value('i', 6)
        self.shared_buffer = torch.empty((self.args.memory, 14115), dtype=torch.float32).share_memory_()
        self.state = self.shared_buffer[:, 0:84 * 84].reshape(-1, 84, 84)
        self.next_state = self.shared_buffer[:, 84 * 84 + 1:2 * 84 * 84 + 1].reshape(-1, 84, 84)
        self.action = self.shared_buffer[:, 84 * 84]
        self.rew = self.shared_buffer[:, 2 * 84 * 84]
        self.done = self.shared_buffer[:, 2 * 84 * 84 + 1]
        self.greedy_actions = greed_act
        self.current_env_step = mp.Manager().Value('i', 0)
        self.act_step = mp.Manager().Value('i', 0)
        self.exp_step = mp.Manager().Value('i', 0)
        self.delay_buffer = mp.Manager().list([0 for _ in range(self.args.num_inf)])
        self.action_queue = mp.Manager().Value('i', 6)
        self.num_noop_actions = mp.Manager().Value('i', 0)
        self.learn_step = mp.Manager().Value('i', 0)
        self.inf_id = mp.Manager().Value('i', 0)
        self.change_count = mp.Manager().Value('i', 0)
        self.t_theta_max = mp.Manager().Value('f', self.args.t_theta_max)
        self.start_time_inf = mp.Manager().Value('f', 0)
        self.inf_learn_step = mp.Manager().Value('f', 0)

        # for regulating the cycle time of the environment thread
        if args.fps > 0.0:
            self.spf = 1.0 / args.fps
            print("spf value:", self.spf)
        else:
            self.spf = 0.0

        if args.aps > 0.0:
            self.spa = 1.0 / args.aps
        else:
            self.spa = 0.0

    def order_distance(self, process_id, curr_process_id):
        if curr_process_id >= process_id:
            return curr_process_id - process_id
        else:
            return self.args.num_inf - (process_id - curr_process_id)

    def inference(self, inf_id):
        print("Inside Inference process, with ID:{}!".format(inf_id))

        if inf_id == 0:
            self.start_time_inf.value = time.time()
            wait_time = 0
        else:
            delay = time.time() - self.start_time_inf.value
            wait_time = (self.t_theta_max.value * inf_id / self.args.num_inf) - delay
        time.sleep(wait_time)

        elapsed_inf = []
        greedy_inf_mean_time = 0
        num_greedy_act = 0

        while self.act_step.value <= int(self.args.total_timesteps):

            current_state = self.state[self.buffer_idx.value].reshape(1, 1, 84, 84)
            self.exp_step.value = min(self.act_step.value, self.current_env_step.value)

            time.sleep(self.delay_buffer[inf_id])

            self.delay_buffer[inf_id] = 0.0

            before_inf = time.time()
            action, greedy_inf_mean_time, num_greedy_act = self.model.getAction(current_state, self.exp_step.value,
                                                                                greedy_inf_mean_time, num_greedy_act)
            after_inf = time.time()
            time_diff = after_inf - before_inf
            delta = time_diff - self.t_theta_max.value
            if delta > 0:
                self.change_count.value += 1
                left_out_idx = [val for val in range(self.args.num_inf) if val != inf_id]
                for idx in left_out_idx:
                    dist = self.order_distance(idx, inf_id)
                    self.delay_buffer[idx] += dist * abs(delta) / self.args.num_inf

                self.t_theta_max.value = time_diff
            else:
                time.sleep(self.t_theta_max.value - time_diff)
            self.act_step.value += 1
            after_inf_sleep = time.time()

            self.action_queue.value = action
            sys.stdout.flush()

            elapsed_inf.append(time_diff)
            if self.act_step.value % 1000 == 0:

                elapsed_inf_mean = np.mean(elapsed_inf)
                elapsed_inf_std = np.std(elapsed_inf)
                elapsed_inf_ste = np.std(elapsed_inf) / np.sqrt(len(elapsed_inf))
                elapsed_inf = []
                # print(" {}: The Mean, std and ste of t_theta:{}, {}, {}".format(mp.current_process().pid, elapsed_inf_mean, elapsed_inf_std, elapsed_inf_ste))
            sys.stdout.flush()

    def learn(self, learn_id):
        print("Inside learn process, with ID:{}!".format(mp.current_process().pid))
        time_learn = []
        while self.learn_step.value <= int(self.args.total_timesteps):
            if self.buffer_idx.value == 0:
                sample_idxs = np.random.randint(1, size=self.args.batch_size)
            else:
                sample_idxs = np.random.randint(min(self.buffer_idx.value, self.args.memory), size=self.args.batch_size)
            sample_state = self.state[sample_idxs].reshape(self.args.batch_size, 1, 84, 84)
            sample_action = self.action[sample_idxs]
            sample_reward = self.rew[sample_idxs]
            sample_nstate = self.next_state[sample_idxs].reshape(self.args.batch_size, 1, 84, 84)
            before_learn = time.time()
            self.model.learn(sample_state, sample_action, sample_reward, sample_nstate, learn_id, self.lock, self.args.device)
            after_learn = time.time()
            self.learn_step.value += 1
            time_1learn = after_learn - before_learn
            time_learn.append(time_1learn)
            if self.learn_step.value % 1000 == 0:
                time_learn_mean = np.mean(time_learn)
                time_learn_std = np.std(time_learn)
                time_learn_ste = np.std(time_learn) / np.sqrt(len(time_learn))
                time_learn = []
                # print(" {}: The Mean, std and ste in Learn Process are:{}, {}, {}".format(mp.current_process().pid, time_learn_mean, time_learn_std, time_learn_ste))
                # print("Current Learn steps and Env steps:", self.learn_step.value, self.current_env_step.value)
            sys.stdout.flush()

    def inflearn(self):
        print("inside inflearn")
        greedy_inf_mean_time = 0
        num_greedy_act = 0

        while self.inf_learn_step.value <= int(self.args.total_timesteps):
            action, greedy_inf_mean_time, num_greedy_act = self.inf_step(self.inf_learn_step.value, greedy_inf_mean_time, num_greedy_act)
            self.action_queue.value = action
            self.learn_step_func()
            self.inf_learn_step.value += 1

    def learninf(self):
        print("inside learninf")

        greedy_inf_mean_time = 0
        num_greedy_act = 0

        while self.inf_learn_step.value <= int(self.args.total_timesteps):
            self.learn_step_func()
            action, greedy_inf_mean_time, num_greedy_act = self.inf_step(self.inf_learn_step.value, greedy_inf_mean_time, num_greedy_act)
            self.action_queue.value = action
            self.inf_learn_step.value += 1

    def inf_step(self, inf_learn_step, greedy_inf_mean_time, num_greedy_act):
        current_state = self.state[self.buffer_idx.value].reshape(1, 1, 84, 84)
        exp_step = min(inf_learn_step, self.current_env_step.value)

        before_inf = time.time()
        action, greedy_inf_mean_time, num_greedy_act = self.model.getAction(current_state, exp_step,
                                                                            greedy_inf_mean_time, num_greedy_act)
        after_inf = time.time()
        return action, greedy_inf_mean_time, num_greedy_act

    def learn_step_func(self):
        if self.buffer_idx.value == 0:
            sample_idxs = np.random.randint(1, size=self.args.batch_size)
        else:
            sample_idxs = np.random.randint(min(self.buffer_idx.value, self.args.memory), size=self.args.batch_size)
        sample_state = self.state[sample_idxs].reshape(self.args.batch_size, 1, 84, 84)
        sample_action = self.action[sample_idxs]
        sample_reward = self.rew[sample_idxs]
        sample_nstate = self.next_state[sample_idxs].reshape(self.args.batch_size, 1, 84, 84)
        before_learn = time.time()
        self.model.learn(sample_state, sample_action, sample_reward, sample_nstate, 0, self.lock)
        self.learn_step.value += 1
        after_learn = time.time()


    def add_to_buffer(self, state, action, reward, next_state):
        self.state[self.buffer_idx.value] = torch.tensor(state).reshape(1, 84, 84)
        self.next_state[self.buffer_idx.value] = torch.tensor(next_state).reshape(1, 84, 84)
        self.rew[self.buffer_idx.value] = reward
        self.done[self.buffer_idx.value] = False
        self.action[self.buffer_idx.value] = action
        self.buffer_idx.value = (self.buffer_idx.value + 1) % self.args.memory

    def environment(self):
        wins = 0
        losses = 0
        episodes = 0
        print("Inside Environment process")
        time_env = []
        time_act = []
        while self.current_env_step.value <= int(self.args.total_timesteps):
            if self.args.game == 'tetris':
                complete = False
                while not complete:
                    level_id = "TypeA_Level" + str(self.args.level)
                    env = gym.make("Tetris-v0", state=level_id, render=False, noops=self.args.noops,
                                   obs_history=self.args.obs_history, fps=0.0)

                    reward = 0.0
                    last_nonzero_reward = 0.0
                    step = 0
                    done = False
                    state = env.reset()
                    while not done:
                        before_env = time.time()
                        step += 1
                        self.state[self.buffer_idx.value] = torch.tensor(state).reshape(1, 84, 84)
                        action = self.action[self.buffer_idx.value].int().numpy()
                        next_state, rew, done, info = env.step(action)
                        after_env = time.time()
                        self.next_state[self.buffer_idx.value] = torch.tensor(next_state).reshape(1, 84, 84)
                        self.rew[self.buffer_idx.value] = rew
                        self.done[self.buffer_idx.value] = done
                        self.buffer_idx.value = (self.buffer_idx.value + 1) % self.args.memory

                        if rew != 0.0:
                            print("something happened:", rew, self.current_env_step.value)
                            last_nonzero_reward = rew
                        reward += rew
                        state = next_state
                        # set cycle time of environment
                        if self.spf > 0.0:
                            elapsed = env_after - env_before
                            diff = self.spf - elapsed
                            if diff > 0.0:
                                time.sleep(diff)

                    if last_nonzero_reward == 1.0:
                        complete = True
                        wins += 1
                    elif last_nonzero_reward == -1.0:
                        losses += 1

                    self.current_env_step.value += step
                    episodes += 1

                    print("episodes=", episodes, "steps=", self.current_env_step.value, "positives=", wins, "negatives=", losses,
                          "ep_reward=", reward)
                    print("epsilon=", self.model.getEpsilon(self.exp_step.value), "greedy_actions=",
                          self.greedy_actions.value)

                    sys.stdout.flush()
                    env.close()

            elif self.args.game in ["red", "blue", "yellow"]:

                for encounter in self.args.encounter_list:
                    complete = False
                    while not complete:

                        env = gym.make("PokemonGen1-v0", state=self.args.default_encounter, render=False,
                                       noops=self.args.noops, obs_history=self.args.obs_history, fps=0.0)
                        reward = 0.0
                        last_nonzero_reward = 0.0
                        step = 0
                        done = False
                        state = env.reset()
                        num_noops = 0
                        reward_sum = 0
                        tmp_action = -1
                        while not done:
                            step += 1
                            before_env = time.time()
                            if step == 1:
                                before_inf = before_env
                            current_action = self.action_queue.value
                            current_act_step = self.act_step.value
                            if current_action != self.action_default.value:

                                next_state, rew, done, info = env.step(current_action)
                                after_env = time.time()
                                after_inf = time.time()
                                time1_act = after_inf - before_inf
                                sys.stdout.flush()
                                before_inf = after_inf
                                time_act.append(time1_act)
                                if current_act_step == self.act_step.value:
                                    self.action_queue.value = self.action_default.value
                                if len(time_act) >= 1000:
                                    time_act_mean = np.mean(time_act)
                                    time_act_std = np.std(time_act)
                                    time_act_ste = np.std(time_act) / np.sqrt(len(time_act))
                                    time_act = []
                                    # print(" {}: The Mean, std and ste of t_i:{}, {}, {}".format(
                                    #     mp.current_process().pid, time_act_mean, time_act_std,
                                    #       time_act_ste))
                                    sys.stdout.flush()

                            else:
                                if self.args.random_not_noop:
                                    random_action = np.random.randint(0, 6)
                                    next_state, rew, done, info = env.step(random_action)
                                    after_env = time.time()
                                else:
                                    next_state, rew, done, info = env.step(current_action)
                                    after_env = time.time()


                            if self.action_queue.value == self.action_default.value:
                                sys.stdout.flush()
                                num_noops += 1
                                tmp_next_state = copy.deepcopy(next_state)
                                reward_sum += rew

                            else:
                                if tmp_action != -1:
                                    self.state[self.buffer_idx.value] = torch.tensor(tmp_state).reshape(1, 84, 84)
                                    self.next_state[self.buffer_idx.value] = torch.tensor(tmp_next_state).reshape(1, 84, 84)
                                    self.action[self.buffer_idx.value] = current_action
                                    if current_action != self.action_default.value:
                                        self.num_noop_actions.value += 1
                                    self.rew[self.buffer_idx.value] = rew
                                    self.done[self.buffer_idx.value] = done
                                    self.buffer_idx.value = (self.buffer_idx.value + 1) % self.args.memory


                                tmp_action = current_action
                                tmp_state = copy.deepcopy(state)
                                reward_sum = rew
                                tmp_next_state = copy.deepcopy(next_state)
                                num_noops = 0

                            if rew != 0.0:
                                print("something happened:", rew, self.current_env_step.value)
                                last_nonzero_reward = rew
                            reward += rew
                            state = next_state

                            # set cycle time of environment
                            if self.spf > 0.0:
                                elapsed = after_env - before_env
                                diff = self.spf - elapsed
                                if diff > 0.0:
                                    time.sleep(diff)
                            after_delay = time.time()
                            time_1env = after_delay - before_env
                            time_env.append(time_1env)
                            if self.current_env_step.value % 1000 == 0:
                                time_env_mean = np.mean(time_env)
                                time_env_std = np.std(time_env)
                                time_env_ste = np.std(time_env) / np.sqrt(len(time_env))
                                time_env = []
                                print(" {}: The Mean, std and ste of t_m:{}, {}, {}".format(
                                    mp.current_process().pid, time_env_mean, time_env_std,
                                      time_env_ste))

                            sys.stdout.flush()
                            if self.current_env_step.value == 20000:
                                print("At 20000 env steps:", "episodes=", episodes, "steps=",
                                      self.current_env_step.value,
                                      "wins=", wins,
                                      "losses=", losses,
                                      "ep_reward=",
                                      reward, "last_encounter=", encounter, "num non-noop action=",
                                      self.num_noop_actions.value,
                                      "num learning step:", self.learn_step.value)
                            self.current_env_step.value += 1
                        if last_nonzero_reward == 1.0:
                            complete = True
                            wins += 1
                        elif last_nonzero_reward == -1.0:
                            losses += 1


                        episodes += 1
                        if self.learn_step.value != 0:
                            print("episodes=", episodes, "steps=", self.current_env_step.value, "wins=", wins, "losses=", losses,
                                  "ep_reward=",
                                  reward, "last_encounter=", encounter, "num non-noop action=", self.num_noop_actions.value,
                                  "num learning step:", self.learn_step.value, "action per frame:",
                                  self.num_noop_actions.value/self.current_env_step.value, "actions per learning:",
                                  self.num_noop_actions.value/(self.args.batch_size * self.learn_step.value))
                            print("epsilon=", self.model.getEpsilon(self.exp_step.value), "greedy_actions=",
                                  self.greedy_actions.value)
                        else:
                            print("episodes=", episodes, "steps=", self.current_env_step.value, "wins=", wins,
                                  "losses=", losses,
                                  "ep_reward=",
                                  reward, "last_encounter=", encounter, "num non-noop action=",
                                  self.num_noop_actions.value,
                                  "num learning step:", self.learn_step.value)
                        print("epsilon=", self.model.getEpsilon(self.exp_step.value), "greedy_actions=",
                              self.greedy_actions.value)

                        sys.stdout.flush()
                        env.close()

            else:
                print('Game not recognized!!!')
        print("FINAL", "wins=", wins, "losses=", losses, "episodes=", episodes, "steps=", self.current_env_step.value)

    def run(self):
        num_processes = self.args.num_inf + self.args.num_learn + 1
        if self.args.mode == "async":
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes, mp_context=torch.multiprocessing) as pool:
                f_env = pool.submit(self.environment)
                f_learn = [pool.submit(self.learn, j) for j in range(self.args.num_learn)]
                f_inf = [pool.submit(self.inference, k) for k in range(self.args.num_inf)]
                print("except f_env:", f_env.exception())
                for e in concurrent.futures.as_completed(f_inf):
                    print("except f_inf:", e.exception())
                for e in concurrent.futures.as_completed(f_learn):
                    print("except f_learn:", e.exception())

        elif self.args.mode == "inflearn":
            with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=torch.multiprocessing) as pool:
                f1 = pool.submit(self.environment)
                f2 = pool.submit(self.inflearn)
                print("except f1:", f1.exception())
                print("except f2:", f2.exception())
        elif self.args.mode == "learninf":
            with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=torch.multiprocessing) as pool:
                f1 = pool.submit(self.environment)
                f2 = pool.submit(self.learninf)
                print("except f1:", f1.exception())
                print("except f2:", f2.exception())
        else:
            print("Error! Mode not recognized:",self.args.mode)

if __name__=='__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Gameboy Experiments!!!')
    parser.add_argument('--game', type=str, default="tetris", help='What game is it: tetris, or Pokemon red, blue or yellow')
    parser.add_argument('--mode', type=str, default="async", help='possibilities: "async" for asynchronous learning and inference threads, "inflearn" for stacked inference before learning in a cycle, and "learninf" for stacked learning before inference in a cycle')

    ### model details
    parser.add_argument('--model', type=str, default="async_model", help='Model for asynchronous learning')
    parser.add_argument('--arch', type=str, default="impala", help='architecture: nature or impala')
    parser.add_argument('--k', type=float, default=16,
                        help='number of filters per layer normalized by standard impala, nature default is 2.0')
    parser.add_argument('--num_inf', type=int, default=1, help='num inference processes')
    parser.add_argument('--num_learn', type=int, default=1, help='num learn processes')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--opt', type=str, default="SharedAdam", help='which optimizer would you like to use? Adam or SGD')
    parser.add_argument('--cuda', type=str, default="no", help='gpu on?')
    parser.add_argument('--explore', type=int, default=100000, help='exploration phase length')
    parser.add_argument('--eps_init', type=float, default=1, help='exploration phase init value')
    parser.add_argument('--eps_end', type=float, default=0.05, help='exploration phase end value')

    parser.add_argument('--target_freq', type=int, default=10000, help='target update frequency (number of updates)')
    parser.add_argument('--n', type=int, default=3, help='n for n-step learning algorithms')
    parser.add_argument('--total_timesteps', type=int, default=1e12, help='lifetime of agent')
    parser.add_argument('--reservoir', type=str, default="yes", help='yes = reservoir sampling, no = recency sampling')

    ### pokemon environment details
    parser.add_argument('--setting', type=str, default="battles", help='battles or catching')
    parser.add_argument('--default_encounter', type=str, default="Charm_B1", help='Default encounter')

    ### tetris environment details
    parser.add_argument('--level', type=int, default=0, help='level number')
    parser.add_argument('--pretrain_steps', type=int, default=0, help='number of pretraining steps')

    ### General environment details
    parser.add_argument('--cluster', type=str, default="ibm", help='ibm or mila')
    parser.add_argument('--passes', type=int, default=1, help='charmander or squirtle or bulbasaur or pikachu')
    parser.add_argument('--fps', type=float, default=59.7275,
                        help='frames per second: 0.0 for no waiting on hardware or 59.7275 for gameboy simulation')
    parser.add_argument('--aps', type=float, default=5.97275,
                        help='actions per second: 0.0 for no waiting on hardware, and 59.7275 for one action each frame')
    parser.add_argument('--noops', type=int, default=0, help='noop actions per actual action')
    parser.add_argument('--num_actions', type=int, default=6, help='number of actions (default is 6)')
    parser.add_argument('--random_not_noop',  action='store_true', help='take random actions instead of noop actions')

    parser.add_argument('--obs_history', type=int, default=1, help='number of observations used in history')
    parser.add_argument('--t_theta_max', type=float, default=0.01, help='t_theta max value for sleeping in inference')


    ### resource constraint details
    parser.add_argument('--memory', type=int, default=10000, help='memory size')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--updates', type=int, default=1, help='updates per step in environment')
    parser.add_argument('--init_step', default=1000, type=int)
    parser.add_argument('--max_update_freq', default=10, type=int)
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.reservoir = True if args.reservoir == 'yes' else False

    args.cuda = False
    args.device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
    print("args:", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.encounter_list = []

    if args.game == 'red':
        if args.setting == "battles":
            args.default_encounter = "Bulb_B1"
            for i in range(284):
                name = "Bulb_B" + str(i + 1)
                args.encounter_list.append(name)

        elif args.setting == "catching":
            args.default_encounter = "Bulb_C1"
            for i in range(91):
                name = "Bulb_C" + str(i + 1)
                args.encounter_list.append(name)

        env = gym.make("PokemonGen1-v0", state=args.default_encounter, render=False, noops=args.noops, obs_history=args.obs_history, fps=0.0)

    elif args.game == 'blue':
        if args.setting == "battles":
            args.default_encounter = "Squirt_B1"
            for i in range(295):
                name = "Squirt_B" + str(i + 1)
                args.encounter_list.append(name)

        elif args.setting == "catching":
            args.default_encounter = "Squirt_C1"
            for i in range(93):
                name = "Squirt_C" + str(i + 1)
                args.encounter_list.append(name)

        env = gym.make("PokemonGen1-v0", state=args.default_encounter, render=False, noops=args.noops,
                       obs_history=args.obs_history, fps=0.0)

    elif args.game == 'yellow':
        if args.setting == "battles":
            args.default_encounter = "Pika_B1"
            for i in range(288):
                name = "Pika_B" + str(i + 1)
                args.encounter_list.append(name)

        elif args.setting == "catching":
            args.default_encounter = "Pika_C1"
            for i in range(92):
                name = "Pika_C" + str(i + 1)
                args.encounter_list.append(name)

        env = gym.make("PokemonGen1-v0", state=args.default_encounter, render=False, noops=args.noops, obs_history=args.obs_history, fps=0.0)


    elif args.game == 'tetris':
        level_id = "TypeA_Level" + str(args.level)
        env = gym.make("Tetris-v0", state=level_id, render=False, noops=args.noops, obs_history=args.obs_history, fps=0.0)

    else:
        print("unrecognized combintation of setting:", args.setting, "and game:", args.game)

    args.state_size = (1, 84, 84)
    mdl = importlib.import_module('learning.' + args.model)
    model = mdl.Net(args)
    model.share_memory()

    exp = Experiment(args, model, model.greedy_actions)

    if args.game == 'tetris' and args.pretrain_steps > 0:
        for episode in range(1):
            if args.cluster == "ibm":
                expert_data_path = 'tetris_episode' + str(episode) + '.pkl'
            else:
                expert_data_path = '/tetris_data/tetris_episode' + str(
                    episode) + '.pkl'

            with open(expert_data_path, 'rb') as f:
                data = pickle.load(f)

            for obs, action, reward, next_obs in data:
                if action < args.num_actions:
                    exp.add_to_buffer(obs, action, reward, next_obs)

        print("offline_data_size=", exp.buffer_idx.value)

        for i in range(args.pretrain_steps):
            exp.learn_step_func()

    print(args.pretrain_steps, "learning steps completed")
    exp.run()
