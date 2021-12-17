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
from wrapper import StochasticFrameSkip, TimeLimit, WarpFrame, ClipRewardEnv, FrameStack, ScaledFloatFrame
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env


def create_env(state):
    env = retro.make('SuperMarioWorld-Snes', state)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=4500)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env


def create_discrete_env(state):
    env = MarioWorldDiscretizer(retro.make('SuperMarioWorld-Snes', state))
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = TimeLimit(env, max_episode_steps=4500)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    return env


def test_random_agent():
    env = create_env('YoshiIsland2')
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
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10_000, log_interval=4)
    model.save('models/qrdqn_mario')


def test_DQN():
    pass


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
    pass
    # TRAIN DQN
    #env = create_discrete_env('YoshiIsland2')
    #model = DQN(policy='MlpPolicy', env=env, batch_size=4, tensorboard_log='./tensorboard/')
    #model.learn(total_timesteps=25_000)
    #model.save('models/dqn_mario')

    # TRAIN PPO
    #env = create_env('YoshiIsland2')
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
