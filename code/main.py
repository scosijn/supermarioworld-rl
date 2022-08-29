import os
import time
import retro
import numpy as np
from wrappers import MarioWrapper
from callbacks import ProgressBar, CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


def make_retro_env(state, n_envs=1):
    """
    Args:
        state (str):
        n_envs (int):
    """

    def mario_wrapper(env):
        env = MarioWrapper(
            env,
            actions=retro.Actions.MULTI_DISCRETE,
            screen_size=(84, 84),
            grayscale=True,
            stickiness=0.25,
            n_skip=4
        )
        env = Monitor(env)
        return env

    env = make_vec_env(
        env_id=retro.RetroEnv,
        n_envs=n_envs,
        wrapper_class=mario_wrapper,
        env_kwargs={
            'game': 'SuperMarioWorld-Snes',
            'state': state,
            'info': './data/data.json',
            'scenario': './data/scenario.json'
        },
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    return env


def PPO_model(env, log='./tensorboard/'):
    model = PPO(policy='CnnPolicy',
                env=env,
                learning_rate=1e-4,
                n_steps=1024,
                batch_size=512,
                n_epochs=2,
                clip_range=0.1,
                ent_coef=0.001,
                tensorboard_log=log)
    return model


def train_model(
    model,
    total_timesteps,
    save_freq=0,
    name_prefix='model',
    reset_num_timesteps=False,
    verbose=0
):
    initial_timesteps = model.num_timesteps
    if reset_num_timesteps:
        initial_timesteps = 0
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./models/',
        name_prefix=name_prefix,
        verbose=verbose
    )
    with ProgressBar(
        initial_timesteps,
        initial_timesteps + total_timesteps
    ) as progress_callback:
        model.learn(
            total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=name_prefix,
            callback=[checkpoint_callback, progress_callback]
        )


def test_model(model, state, record=False, record_path='./recordings/'):
    env = make_retro_env(state)
    obs = env.reset()
    if record:
        os.makedirs(record_path, exist_ok=True)
        env.env_method('record_movie', f'{record_path}{state}.bk2')
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.01)
    env.close()


def random_agent(env):
    """
    Agent that will take a random action on each timestep.

    Args:
        env (Gym Environment): the environment #TODO VECENV
    """
    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(np.tile(action, (env.num_envs, 1)))
        env.render()


def main():
    env = make_retro_env('YoshiIsland1', n_envs=8)
    model = PPO_model(env)
    train_model(model,
                total_timesteps=25_000_000,
                name_prefix='PPO_YoshiIsland1_1024')
    model.save('./models/PPO_YoshiIsland1_1024')
    env.close()


if __name__ == '__main__':
    main()
