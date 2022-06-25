import gym
import retro
import numpy as np
from collections import deque
from gym import spaces
from gym.wrappers import ResizeObservation
from gym.wrappers import GrayScaleObservation


class MultiDiscreteActions(gym.ActionWrapper):
    """
    Wrap a gym environment and convert from MultiBinary to MultiDiscrete action space.

    Args:
        env (Gym Environment): the environment to wrap
        actions: list of lists of valid actions where each list represents a discrete space
    """

    def __init__(self, env, actions):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary)
        assert isinstance(env.action_space.n, int)
        self._convert_action = []
        self._n = env.action_space.n
        buttons = env.unwrapped.buttons
        for action in actions:
            action_map = dict()
            for i, button in enumerate(action, start=1):
                action_map[i] = buttons.index(button)
            self._convert_action.append(action_map)
        self.action_space = spaces.MultiDiscrete([len(x) + 1 for x in actions])
        self.use_restricted_actions = retro.Actions.MULTI_DISCRETE

    def action(self, action):
        converted_action = np.full(self._n, False)
        for i, v in enumerate(action):
            if v > 0: # 0 = no action
                converted_action[self._convert_action[i][v]] = True
        return converted_action


class SuperMarioWorldActions(MultiDiscreteActions):
    """
    Convert actions for the SNES game Super Mario World to a MultiDiscrete action space.

    Args:
        env (Gym Environment): the environment to wrap
    """

    def __init__(self, env):
        actions = [
            ['UP', 'DOWN', 'LEFT', 'RIGHT'], # arrow keys
            ['A', 'B'],                      # jump keys
            ['X']                            # special keys
        ]
        super().__init__(env, actions)


class RandomStart(gym.Wrapper):
    """
    Randomize the environment by waiting a random number of steps on reset.

    Args:
        env (Gym Environment): the environment to wrap
        max_steps (int): the maximum amount of steps to wait
    """

    def __init__(self, env, max_steps=30):
        super().__init__(env)
        self.max_steps = max_steps

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        steps = self.unwrapped.np_random.randint(1, self.max_steps + 1)
        wait = np.zeros(self.env.action_space.shape, dtype=int)
        obs = None
        for _ in range(steps):
            obs, _, done, _ = self.env.step(wait)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class ResetOnLifeLost(gym.Wrapper):
    """
    Reset the environment when a life is lost.

    Args:
        env (Gym Environment): the environment to wrap
    """

    def __init__(self, env):
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


class FrameSkip(gym.Wrapper):
    """
    Return only every n-th frame.

    Args:
        env (Gym Environment): the environment to wrap
        n_skip (int): the number of frames to skip
    """

    def __init__(self, env, n_skip=4):
        super().__init__(env)
        self.n_skip = n_skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.n_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque([], maxlen=n_stack)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], shape[2] * n_stack),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_stack
        return np.concatenate(self.frames, axis=2)


class FrameStackLazy(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = deque([], maxlen=n_stack)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(shape[0], shape[1], shape[2] * n_stack),
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.n_stack):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_stack
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


def wrap_env(env):
    env = SuperMarioWorldActions(env)
    env = ResetOnLifeLost(env)
    env = RandomStart(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    env = FrameSkip(env, n_skip=4)
    env = FrameStack(env, n_stack=4)
    return env
