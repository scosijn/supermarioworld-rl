from cgitb import reset
import gym
import retro
import itertools
import numpy as np
from collections import deque
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
    def __init__(self, env, min_reward=-15, max_reward=15):
        super().__init__(env)
        self.prev_x_pos = None
        self.checkpoint = None
        self.min_reward = min_reward
        self.max_reward = max_reward
        self._reward_range = (min_reward, max_reward)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = None
        self.checkpoint = None
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs, self.reward(rew, info), done, info

    def reward(self, rew, info):
        x_pos = info['x_pos']
        checkpoint = info['checkpoint']
        endoflevel = info['endoflevel']
        is_dying = info['is_dying']
        if self.prev_x_pos is None:
            self.prev_x_pos = x_pos
        if self.checkpoint is None:
            self.checkpoint = checkpoint
        rew += x_pos - self.prev_x_pos
        self.prev_x_pos = x_pos
        if self.checkpoint == 0 and checkpoint == 1:
            self.checkpoint = checkpoint
            rew += (self.max_reward / 2)
        if endoflevel == 1:
            rew += self.max_reward
        if is_dying == 1:
            rew -= self.max_reward
        return np.clip(rew, self.min_reward, self.max_reward)


#class RandomStart(gym.Wrapper):
#    """
#    Randomize the environment by waiting a random number of steps on reset.
#
#    Args:
#        env (Gym Environment): the environment to wrap
#        max_steps (int): the maximum amount of steps to wait
#    """
#
#    def __init__(self, env, max_steps=30):
#        super().__init__(env)
#        assert max_steps >= 0
#        self.max_steps = max_steps
#
#    def reset(self, **kwargs):
#        self.env.reset(**kwargs)
#        steps = self.unwrapped.np_random.randint(1, self.max_steps + 1)
#        wait = np.zeros(self.env.action_space.shape, dtype=int)
#        obs = None
#        for _ in range(steps):
#            obs, _, done, _ = self.env.step(wait)
#            if done:
#                obs = self.env.reset(**kwargs)
#        return obs


#class ResetOnLifeLost(gym.Wrapper):
#    """
#    Reset the environment when a life is lost.
#
#    Args:
#        env (Gym Environment): the environment to wrap
#    """
#
#    def __init__(self, env):
#        super().__init__(env)
#        self.max_lives = -1
#    
#    def step(self, action):
#        obs, reward, done, info = self.env.step(action)
#        lives = info['lives']
#        if self.max_lives == -1:
#            self.max_lives = lives
#        elif lives < self.max_lives:
#            done = True
#            info['lost_life'] = True
#        return obs, reward, done, info
#
#    def reset(self, **kwargs):
#        obs = self.env.reset(**kwargs)
#        self.max_lives = -1
#        return obs


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


class MarioWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        actions=retro.Actions.ALL,
        screen_size=None,
        grayscale=False,
        stickiness=0,
        n_skip=1,
        rewards=None
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
        if rewards is not None:
            min, max = rewards
            env = MarioRewardWrapper(env, min_reward=min, max_reward=max)
        super().__init__(env)


def wrap_env(env):
    env = MarioActionWrapper(env, retro.Actions.MULTI_DISCRETE)
    env = ResizeObservation(env, shape=(84, 84))
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResetOnLifeLost(env)
    env = StickyActions(env, stickiness=0.25)
    env = FrameSkip(env, n_skip=4)
    env = FrameStack(env, n_stack=4)
    env = MarioRewardWrapper(env, min_reward=-15, max_reward=15)
    return env
