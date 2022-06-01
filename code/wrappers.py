import gym
import retro
import itertools
import cv2
import numpy as np
from gym import spaces
from collections import deque


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Source: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py

    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = spaces.Discrete(len(self._decode_discrete_action))
        self.use_restricted_actions = retro.Actions.DISCRETE

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class MarioWorldDiscretizer(Discretizer):
    def __init__(self, env):
        """
        Convert actions for the SNES game Super Mario World to a discrete space.

        :param env: (Gym Environment) the environment to wrap
        """
        combos = []
        arrow_keys = [None, 'UP', 'DOWN', 'LEFT', 'RIGHT']
        jump_keys = [None, 'A', 'B']
        special_keys = [None, 'X']
        for combo in itertools.product(arrow_keys, jump_keys, special_keys):
            combos.append(list(filter(None, combo)))
        super().__init__(env, combos)


class NoopReset(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.

        :param env: (Gym Environment) the environment to wrap
        :param noop_max: (int) the maximum value of no-ops to run
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLife(gym.Wrapper):
    def __init__(self, env):
        """
        Make end-of-life == end-of-episode

        :param env: (Gym Environment) the environment to wrap
        """
        super().__init__(env)
        self.max_lives = -1
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        lives = info['lives']
        if self.max_lives == -1:
            self.max_lives = lives
        elif lives < self.max_lives:
            done = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.max_lives = -1
        return obs


class ResizeFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and warp frames to 84x84 (default)

    :param env: (Gym Environment) the environment
    :param width: (int) screen width in pixels
    :param height: (int) screen height in pixels
    :param gray: (bool) convert to grayscale
    """
    def __init__(self, env, width=84, height=84, gray=True):
        super().__init__(env)
        self.screen_size = (width, height)
        self.gray = gray
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(width, height, 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: environment frame
        :return: the observation
        """
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, self.screen_size, interpolation=cv2.INTER_AREA)
        if self.gray:
            return frame[:, :, None]
        return frame[:, :, :]


class SkipFrame(gym.Wrapper):
    def __init__(self, env, frame_skips=4):
        """
        Return only every n-th frame

        :param env: (Gym Environment) the environment
        :param skip: (int) number of frame skips
        """
        super().__init__(env)
        self._frame_skips = frame_skips

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: the action
        :return: observation, reward, done, information
        """
        total_reward = 0.0
        done = False
        for _ in range(self._frame_skips):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class StackFrame(gym.Wrapper):
    def __init__(self, env, frame_stack=4):
        """
        Stack frame_stack last frames.

        :param env: (Gym Environment) the environment
        :param frame_stack: (int) the number of frames to stack
        """
        super().__init__(env)
        self.frame_stack=frame_stack
        self.frames = deque([], maxlen=frame_stack)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * frame_stack),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.frame_stack):
            self.frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self.frames) == self.frame_stack
        return np.concatenate(self.frames, axis=2)


def wrap_env(env):
    env = MarioWorldDiscretizer(env)
    env = NoopReset(env)
    env = SkipFrame(env)
    env = EpisodicLife(env)
    env = ResizeFrame(env)
    env = StackFrame(env)
    return env
