import gym
import retro
import time
import numpy as np
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from discretizer import MarioWorldDiscretizer
from monitor import MarioWorldMonitor
from wrappers import wrap_env
from callbacks import ProgressBar
#A      = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # spin
#B      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # jump
#X      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] # run
#UP     = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#DOWN   = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#LEFT   = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#RIGHT  = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def create_env(state, is_eval=False):
    env = retro.RetroEnv(
        game='SuperMarioWorld-Snes',
        state=state,
        info='./data/data.json',
        scenario='./data/scenario.json',
    )
    # record the agent's actions if evaluating
    if is_eval:
        env.auto_record('./playback/')
    # convert to discrete action space
    env = MarioWorldDiscretizer(env)
    # check if env is correct according to gym API
    check_env(env)
    return env


def print_env_info(env):
    print('system: {}'.format(env.system))
    print('gamename: {}'.format(env.gamename))
    print('statename: {}'.format(env.statename))
    print('buttons: {}'.format(env.buttons))
    print('action_space: {}'.format(env.action_space))
    print('observation_space: {}'.format(env.observation_space.shape))
    print('reward_range: {}'.format(env.reward_range))


def test_random_agent(env):
    env.reset()
    while True:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()


def train_model(model, total_timesteps, eval_env=None, eval_freq=10000):
    eval_callback = None
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            eval_freq=eval_freq,
            best_model_save_path='./logs/best_model',
            log_path='./logs/results'
        )
    with ProgressBar(total_timesteps) as progress_callback:
        callback = [progress_callback]
        if eval_callback is not None:
            callback.append(eval_callback)
        model.learn(total_timesteps, callback=callback)


def train_DQN(env, model_name, total_timesteps):
    policy_kwargs = dict(n_quantiles=50)
    model = DQN('MlpPolicy', env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=10_000, log_interval=4)
    model.save('models/qrdqn_mario')


def test_DQN(env, model_name):
    return


def train_PPO(env, model_name, total_timesteps):
    model = PPO(policy='CnnPolicy',
                env=env,
                learning_rate=lambda f : f * 2.5e-4,
                n_steps=128,
                batch_size=4,
                n_epochs=4,
                ent_coef=.01,
                clip_range=0.1,
                gamma=0.99,
                gae_lambda=0.95,
                tensorboard_log='./tensorboard/')
    model.learn(total_timesteps=total_timesteps)
    model.save('./models/{}'.format(model_name))


def test_PPO(env, model_name):
    model = PPO.load('./models/{}'.format(model_name))
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()


def main():
    STATE = 'YoshiIsland2'
    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')
    model = PPO(policy='MlpPolicy',
                env=env,
                n_steps=32,
                n_epochs=20,
                batch_size=256,
                ent_coef=0.0,
                gae_lambda=0.8,
                gamma=0.98,
                learning_rate=lambda f : f * 0.001,
                clip_range=lambda f : f * 0.2)
    train_model(model, 20_000, eval_env)
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
