import math
import gym
from gym import error, utils
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
import numpy as np
import time
import math
import copy
from gym import spaces
import retro
import os
import cv2


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


class gen1_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state="Squirt_B1", noops=250, obs_history=1, render=False, fps=0.0):
        self.noops = noops
        self.action_space = spaces.Discrete(6)
        SCRIPT_DIR = os.getcwd()
        retro.data.Integrations.add_custom_path(os.path.join(SCRIPT_DIR, "custom_integrations"))
        if "Bulb_B" in state:
            env = retro.make("PokemonRed_Battles", state=state, inttype=retro.data.Integrations.ALL)
        elif "Bulb_C" in state:
            env = retro.make("PokemonRed_Catching", state=state, inttype=retro.data.Integrations.ALL)
        elif "Squirt_B" in state:
            env = retro.make("PokemonBlue_Battles", state=state, inttype=retro.data.Integrations.ALL)
        elif "Squirt_C" in state:
            env = retro.make("PokemonBlue_Catching", state=state, inttype=retro.data.Integrations.ALL)
        elif "Pika_B" in state:
            env = retro.make("PokemonYellow_Battles", state=state, inttype=retro.data.Integrations.ALL)
        elif "Pika_C" in state:
            env = retro.make("PokemonYellow_Catching", state=state, inttype=retro.data.Integrations.ALL)
        else:
            print("ERROR:", "state not recognized", state)

        env = WarpFrame(env)
        self.env = env
        self.observation_space = self.env.observation_space
        self.steps = 0
        self.render = render
        if fps > 0.0:
            self.spf = 1.0 / fps
        else:
            self.spf = 0.0

        # print("obs=",self.observation_space)
        self.obs_history = obs_history
        self.history = []
        for i in range(self.obs_history):
            self.history.append(np.zeros(shape=(84, 84, 1)))

    def getObservation(self, obs):
        self.history = self.history[1:]
        self.history.append(obs)
        observation = np.concatenate(self.history, axis=2)
        # print("shape=",observation.shape,observation.sum(axis=0).sum(axis=0))
        return observation

    def env_step(self, action):
        # buttons= ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
        if self.spf > 0.0:
            begin = time.time()

        if action == 0:
            myaction = [0, 0, 0, 0, 0, 0, 0, 0, 1]  # A
        elif action == 1:
            myaction = [1, 0, 0, 0, 0, 0, 0, 0, 0]  # B
        elif action == 2:
            myaction = [0, 0, 0, 0, 0, 0, 1, 0, 0]  # LEFT
        elif action == 3:
            myaction = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # UP
        elif action == 4:
            myaction = [0, 0, 0, 0, 0, 0, 0, 1, 0]  # RIGHT
        elif action == 5:
            myaction = [0, 0, 0, 0, 0, 1, 0, 0, 0]  # DOWN
        elif action == 6:
            myaction = [0, 1, 0, 0, 0, 0, 0, 0, 0]  # NOOP
        else:
            print("action is:", action)

            print("ERROR!", "Action not recognized!")

        obs, rew, done, inf = self.env.step(myaction)

        if self.spf > 0.0:
            end = time.time()
            elapsed = end - begin
            diff = self.spf - elapsed
            if diff > 0.0:
                time.sleep(diff)

        return obs, rew, done, inf

    def step(self, action):
        obs, rew, done, inf = self.env_step(action)

        reward = rew
        observation = obs
        self.done = done
        info = inf
        if not self.done:
            for i in range(self.noops):
                obs, rew, done, inf = self.env_step(6)
                if self.render:
                    self.env.render()
                reward += rew
                observation = obs
                self.done = done
                info = inf

        full_observation = self.getObservation(observation)
        if self.render:
            self.env.render()

        return full_observation, reward, self.done, info

    def close(self):
        self.env.close()

    def reset(self):
        if self.spf > 0.0:
            begin = time.time()
        self.done = False
        obs = self.env.reset()
        full_observation = self.getObservation(obs)

        if self.spf > 0.0:
            end = time.time()
            elapsed = end - begin
            diff = self.spf - elapsed
            if diff > 0.0:
                time.sleep(diff)

        return full_observation


register(
    id='PokemonGen1-v0',
    entry_point='environments.pokemon_gen1_env:gen1_env',
)
