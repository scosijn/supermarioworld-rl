import gym
import retro
import itertools
import numpy as np
from collections import deque
from gym import spaces
from gym.wrappers import ResizeObservation
from gym.wrappers import GrayScaleObservation


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Source: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py

    Args:
        env: the environment to wrap
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
    """
    Convert actions for the SNES game Super Mario World to a discrete space.

    Args:
        env (Gym Environment): the environment to wrap
    """

    def __init__(self, env):
        combos = []
        arrow_keys = [None, 'UP', 'DOWN', 'LEFT', 'RIGHT']
        jump_keys = [None, 'A', 'B']
        special_keys = [None, 'X']
        for combo in itertools.product(arrow_keys, jump_keys, special_keys):
            combos.append(list(filter(None, combo)))
        super().__init__(env, combos)


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
            ['UP', 'DOWN', 'LEFT', 'RIGHT'],
            ['A', 'B', 'X'],
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
        assert max_steps >= 0
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
    Return only every `n_skip` frames.

    Args:
        env (Gym Environment): the environment to wrap
        n_skip (int): the number of frames to skip
    """

    def __init__(self, env, n_skip=4):
        super().__init__(env)
        assert n_skip > 0
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
    """
    Stack the last `n_stack` observations.

    Args:
        env (Gym Environment): the environment to wrap
        n_stack (int): the number of observations to stack
    """

    def __init__(self, env, n_stack=4):
        super().__init__(env)
        assert n_stack > 0
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
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        return np.concatenate(self.frames, axis=2)


class StickyActions(gym.Wrapper):
    """
    Sticky actions are used to introduce stochasticity into the environment.
    At every time step there is a chance that the agent will repeat its previous action.

    Args:
        env (Gym Environment): the environment to wrap
        stickiness (float): the probability of executing the previous action
    """
    def __init__(self, env, stickiness):
        super().__init__(env)
        assert 0 <= stickiness <= 1
        self.stickiness = stickiness

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.rng = np.random.default_rng()
        self.previous_action = None
        return obs

    def step(self, action):
        if (
            self.previous_action is not None
            and self.rng.uniform() < self.stickiness
        ):
            action = self.previous_action
        self.previous_action = action
        return self.env.step(action)


def wrap_env(env):
    env = SuperMarioWorldActions(env)
    #env = MarioWorldDiscretizer(env)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResetOnLifeLost(env)
    env = StickyActions(env, stickiness=0.25)
    env = FrameSkip(env, n_skip=4)
    env = FrameStack(env, n_stack=4)
    return env
