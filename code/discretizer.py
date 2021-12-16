"""
Define discrete action spaces for Gym Retro environments with a limited set of button combos
Based on https://github.com/openai/retro/blob/master/retro/examples/discretizer.py
"""

import gym
import itertools
import numpy as np
from gym import spaces


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

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

    def action(self, act):
        return self._decode_discrete_action[act].copy()


class MarioWorldDiscretizer(Discretizer):
    """
    Convert actions for the SNES game Super Mario World to a discrete space.
    """

    def __init__(self, env):
        combos = []
        arrow_keys = [None, 'UP', 'DOWN', 'LEFT', 'RIGHT']
        jump_keys = [None, 'A', 'B']
        special_keys = [None, 'X']
        for combo in itertools.product(arrow_keys, jump_keys, special_keys):
            combos.append(list(filter(None, combo)))
        super().__init__(env, combos)
