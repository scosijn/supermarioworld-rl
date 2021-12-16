import gym
import retro
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Video
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from callback import SaveBestCallback, ProgressBarManager, VideoRecorderCallback
from video import record_video, show_video
from discretizer import MarioWorldDiscretizer
from monitor import MarioWorldMonitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import QRDQN

import argparse
import cv2
from collections import deque
from gym import spaces

#action_b      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # B
#action_y      = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # Y
#action_blank1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] # SELECT
#action_blank2 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] # START
#action_up     = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # UP
#action_down   = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] # DOWN
#action_left   = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] # LEFT
#action_right  = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] # RIGHT
#action_a      = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # A
#action_x      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] # X
#action_l      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] # L
#action_r      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # R


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        done = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i==0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i==1:
                self.curac = ac
            if self.supports_want_render and i<self.n-1:
                ob, rew, done, info = self.env.step(self.curac, want_render=False)
            else:
                ob, rew, done, info = self.env.step(self.curac)
            totrew += rew
            if done: break
        return ob, totrew, done, info

    def seed(self, s):
        self.rng.seed(s)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
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

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


def make_retro():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2')
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=4500)
    return env


def make_retro_disc():
    env = MarioWorldDiscretizer(retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2'))
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=4500)
    return env


def wrap_deemind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env


def create_env():
    env = retro.make(game='SuperMarioWorld-Snes')
    env = MarioWorldDiscretizer(env)
    return env


def create_dqn_env():
    env = retro.make(game='SuperMarioWorld-Snes')
    env = MarioWorldDiscretizer(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    return env


def test_random_agent():
    env = create_env()
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


def train_DQN():
    env = MarioWorldDiscretizer(retro.make(game='SuperMarioWorld-Snes'))
    policy_kwargs = dict(n_quantiles=50)
    model = QRDQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10_000, log_interval=4)
    model.save('models/qrdqn_mario')


def train_PPO():
    env = retro.make(game='SuperMarioWorld-Snes', state='YoshiIsland2')
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./ppo_mario_tensorboard/')
    model.learn(total_timesteps=500)
    model.save('models/ppo_mario')


def test_PPO(model_name):
    model = PPO.load('models/' + model_name)
    env = model.get_env()
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()


def evaluate(model, env, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    all_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
        all_rewards.append(sum(episode_rewards))
    mean_reward = np.mean(all_rewards)
    print("mean_reward:", mean_reward, "num_episodes:", num_episodes)
    #evaluate(model, env, num_episodes=100)
    #mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    #print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward


def multi_process_test():
    env_id = 'CartPole-v1'
    PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
    NUM_EXPERIMENTS = 3
    TRAIN_STEPS = 5000
    EVAL_EPS = 20
    ALGO = PPO
    eval_env = gym.make(env_id)
    reward_averages = []
    reward_std = []
    training_times = []
    total_procs = 0
    for n_procs in PROCESSES_TO_TEST:
        total_procs += n_procs
        print('Running for n_procs = {}'.format(n_procs))
        train_env = make_vec_env(env_id, n_procs)
        rewards = []
        times = []
        for experiment in range(NUM_EXPERIMENTS):
            train_env.reset()
            model = ALGO('MlpPolicy', train_env, verbose=0)
            start = time.time()
            model.learn(total_timesteps=TRAIN_STEPS)
            times.append(time.time() - start)
            mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
            rewards.append(mean_reward)
        train_env.close()
        reward_averages.append(np.mean(rewards))
        reward_std.append(np.std(rewards))
        training_times.append(np.mean(times))
    training_steps_per_second = [TRAIN_STEPS / t for t in training_times]
    plt.figure(figsize=(9, 4))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(1, 2, 1)
    plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
    plt.xlabel('Processes')
    plt.ylabel('Average return')
    plt.subplot(1, 2, 2)
    plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
    plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
    plt.xlabel('Processes')
    plt.ylabel('Training steps per second')
    plt.show()


def callback_chain_test():
    n_steps = 5000
    env_id = 'CartPole-v1'
    env = gym.make(env_id)
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/best_model',
                                 log_path='./logs/results', eval_freq=500)
    model = PPO('MlpPolicy', env)
    with ProgressBarManager(n_steps) as progress_callback:
        model.learn(n_steps, callback=[progress_callback, eval_callback])


def print_env_info(env):
    print('action_space: {}'.format(env.action_space))
    print('num_buttons: {}'.format(env.num_buttons))
    print('buttons: {}'.format(env.buttons))
    print('button_combos: {}'.format(env.button_combos))
    print('use_restricted_actions: {}'.format(env.use_restricted_actions))
    print('observation_space: {}'.format(env.observation_space.shape))
    print('reward_range: {}'.format(env.reward_range))


def main():
    env = create_dqn_env()
    model = DQN(policy='MlpPolicy', env=env, batch_size=4, tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=25_000)
    model.save('models/dqn_mario')
    #env = make_retro_disc()
    #env = wrap_deemind_retro(env)

    # TRAIN
    #model = PPO(policy='MlpPolicy',
    #            env=env,
    #            learning_rate=lambda f : f * 2.5e-4,
    #            n_steps=128,
    #            batch_size=4,
    #            n_epochs=4,
    #            ent_coef=.01,
    #            clip_range=0.1,
    #            gamma=0.99,
    #            gae_lambda=0.95,
    #            tensorboard_log='./tensorboard/')
    #model.learn(total_timesteps=25_000)
    #model.save('models/ppo_mario_disc')

    # TEST
    #model = PPO.load('./models/ppo_mario_disc')
    #obs = env.reset()
    #while True:
    #    time.sleep(0.0075)
    #    action, _states = model.predict(obs)
    #    obs, reward, done, info = env.step(action)
    #    env.render()
    #    if done:
    #        obs = env.reset()


if __name__ == '__main__':
    main()
