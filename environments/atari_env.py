import math
import gymnasium as gym
import ale_py
from gymnasium import error, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import numpy as np
import time
import math
import copy
from gymnasium import spaces
import os
import cv2
import random

def white(my_observation):
    background = my_observation[0][0][0]
    dela_observation = my_observation - background
    sumz = dela_observation.sum()
    return sumz == 0

def same(current_obs,ref_obs):
    delta_obs = current_obs - ref_obs
    sumz = delta_obs.sum()
    return sumz == 0

class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)
    as done in the Nature paper and later work.
    :param env: Environment to wrap
    :param width: New frame width
    :param height: New frame height
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=env.observation_space.dtype
        )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        returns the current observation from a frame
        :param frame: environment frame
        :return: the observation
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class atari(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,state="PongNoFrameskip-v4",noops=0,obs_history=1,render=False,fps=0.0):
        self.noops = noops
        if "Pong" in state:
            self.action_space = spaces.Discrete(6)
        elif "Breakout" in state:
            self.action_space = spaces.Discrete(4)
        elif "Boxing" in state:
            self.action_space = spaces.Discrete(18)
        elif "Krull" in state:
            self.action_space = spaces.Discrete(18)
        elif "Demon" in state:
            self.action_space = spaces.Discrete(6)
        elif "CrazyClimber" in state:
            self.action_space = spaces.Discrete(9)
        elif "Name" in state:
            self.action_space = spaces.Discrete(6)
        else:
            print("Game Not Recognized!",state)
            print(fail)

        env = gym.make(state)
        env = WarpFrame(env)


        # obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        self.env = env
        self.observation_space = self.env.observation_space
        self.steps = 0
        self.render = render
        # self.score_bcd = 0.0

        #print("obs=",self.observation_space)
        self.obs_history = obs_history
        self.history = []
        for i in range(self.obs_history):
            self.history.append(np.zeros(shape=(84,84,1)))

        if fps > 0.0:
            self.spf = 1.0/fps
        else:
            self.spf = 0.0




    def getObservation(self,obs):
        self.history = self.history[1:]
        self.history.append(obs)
        observation = np.concatenate(self.history,axis=2)
        # print("shape=",observation.shape,observation.sum(axis=0).sum(axis=0))
        return observation

    def step(self, action):
        if self.spf > 0.0:
            begin = time.time()

        if self.done:
            print("ERROR!","Must reset environment after episode!")
        else:
            obs, rew, done, truncated, inf = self.env.step(action)
            if self.render:
                self.env.render()
                
            reward = rew
            observation = obs
            self.done = done
            info = inf
            noaction = 0
            for i in range(self.noops):
                obs, rew, done, truncated, inf = self.env.step(noaction)
                if self.render:
                    self.env.render()
                reward += rew
                observation = obs
                self.done = done
                info = inf

        full_observation = self.getObservation(observation)
        # reward = float(info["score_bcd"]) - self.score_bcd
        # self.score_bcd = float(info["score_bcd"])
        if self.spf > 0.0:
            end = time.time()
            elapsed = end-begin
            diff = self.spf-elapsed
            if diff > 0.0:
                time.sleep(diff)

        return full_observation, reward, self.done, info

    def close(self):
        self.env.close()

    def reset(self):
        if self.spf > 0.0:
            begin = time.time()

        # self.score_bcd = 0.0
        self.done = False
        obs, info = self.env.reset()
        full_observation = self.getObservation(obs)

        if self.spf > 0.0:
            end = time.time()
            elapsed = end-begin
            diff = self.spf-elapsed
            if diff > 0.0:
                time.sleep(diff)

        return full_observation





register(
    id='Atari-v0',
    entry_point='environments.atari_env:atari',
)
