import gym
import retro
import itertools
import numpy as np
from gym import spaces
from gym.wrappers import ResizeObservation
from gym.wrappers import GrayScaleObservation


class Discretizer():
    """
    Wrap a gym environment and make it use Discrete actions.
    Source: https://github.com/openai/retro/blob/master/retro/examples/discretizer.py

    Args:
        env (Gym Environment): the environment to wrap
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)
        self.action_space = spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class MultiDiscretizer():
    """
    Wrap a gym environment and make it use MultiDiscrete actions.

    Args:
        env (Gym Environment): the environment to wrap
        actions: list of lists of valid actions where each list represents a discrete space
    """

    def __init__(self, env, actions):
        self._convert_action = []
        self._n = env.action_space.n
        buttons = env.unwrapped.buttons
        for action in actions:
            action_map = dict()
            for i, button in enumerate(action, start=1):
                action_map[i] = buttons.index(button)
            self._convert_action.append(action_map)
        self.action_space = spaces.MultiDiscrete([len(x) + 1 for x in actions])

    def action(self, act):
        converted_action = np.full(self._n, False)
        for i, v in enumerate(act):
            if v > 0: # 0 = no action
                converted_action[self._convert_action[i][v]] = True
        return converted_action


class MarioActionWrapper(gym.ActionWrapper):
    """
    Convert actions for the game Super Mario World to a Discrete or MultiDiscrete action space.

    Args:
        env (Gym Environment): the environment to wrap
        aspace (retro.Actions): action space to convert to
    """

    def __init__(self, env, aspace):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiBinary)
        assert isinstance(env.action_space.n, int)
        assert (aspace == retro.Actions.DISCRETE or
                aspace == retro.Actions.MULTI_DISCRETE)
        arrow_keys = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        special_keys = ['A', 'B', 'X']
        if aspace == retro.Actions.DISCRETE:
            combos = []
            for combo in itertools.product([None] + arrow_keys, [None] + special_keys):
                combos.append(list(filter(None, combo)))
            self.action_wrapper = Discretizer(env, combos)
        elif aspace == retro.Actions.MULTI_DISCRETE:
            self.action_wrapper = MultiDiscretizer(env, [arrow_keys, special_keys])
        self.action_space = self.action_wrapper.action_space
        self.use_restricted_actions = aspace

    def action(self, act):
        return self.action_wrapper.action(act)


class MarioRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, include_y_pos=False):
        super().__init__(env)
        self.include_y_pos = include_y_pos
        self.prev_x_pos = None
        self.prev_y_pos = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = None
        self.prev_y_pos = None
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.reward(rew, info), done, info

    def reward(self, rew, info):
        rew = -1
        x_pos = info['x_pos']
        if self.prev_x_pos is None:
            self.prev_x_pos = x_pos
        rew += x_pos - self.prev_x_pos
        self.prev_x_pos = x_pos
        if self.include_y_pos:
            y_pos = info['y_pos']
            if self.prev_y_pos is None:
                self.prev_y_pos = y_pos
            rew += np.sign(-1 * (y_pos - self.prev_y_pos))
            self.prev_y_pos = y_pos
        return rew


class ResetOnLifeLost(gym.Wrapper):
    """
    Send done signal when a life is lost.

    Args:
        env (Gym Environment): the environment to wrap
    """

    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['animation'] == 9:
            info['is_dying'] = 1
            done = True
        else:
            info['is_dying'] = 0
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


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


class MarioWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        actions=retro.Actions.ALL,
        screen_size=None,
        grayscale=False,
        stickiness=0,
        n_skip=1,
        y_pos_reward=False
    ):
        if actions != retro.Actions.ALL:
            env = MarioActionWrapper(env, actions)
        if screen_size is not None:
            env = ResizeObservation(env, screen_size)
        if grayscale:
            env = GrayScaleObservation(env, keep_dim=True)
        env = ResetOnLifeLost(env)
        env = StickyActions(env, stickiness)
        env = FrameSkip(env, n_skip)
        env = MarioRewardWrapper(env, y_pos_reward)
        super().__init__(env)
