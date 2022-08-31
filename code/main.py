import os
import time
import retro
import numpy as np
from wrappers import MarioWrapper
from callbacks import ProgressBar, CheckpointCallback
from utils import recording_to_video, plot_rollout
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env


def make_retro_env(state, n_envs=1):
    """
    Create a vectorized Gym Retro environment for the game SuperMarioWorld.
    Runs `n_envs` environments in parallel, each in its own process.
    The argument `state` is the name of the level to create the environment(s) for. 
    See https://github.com/openai/retro/tree/master/retro/data/stable/SuperMarioWorld-Snes
    for an overview of all the states included in Gym Retro.

    Args:
        state (str): name of the state (level of the game)
        n_envs (int): the number of environments to run in parallel
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
    """
    Create a reinforcement learning model which will use Proximal Policy Optimization (PPO)
    to learn from the specified environment.

    Args:
        env (Gym Environment): the environment to learn from
        log (str): the log location for tensorboard
    """

    model = PPO(policy='CnnPolicy',
                env=env,
                learning_rate=1e-4,
                n_steps=512,
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
    """
    Train a model for `total_timesteps` timesteps.

    Args:
        model: the model to train
        total_timesteps (int): the number of timesteps to train for
        save_freq (int): the number of steps before saving a checkpoint
        name_prefix (str): prefix to use for the model name if saving
        reset_num_timesteps (bool): if true, reset num_timesteps when training a loaded model
        verbose (int): verbosity level, <= 0: no output, > 0: info
    """

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
    """
    Test a model one some level of the game by playing an episode.

    Args:
        model: the model to test
        state (str): name of the state (level of the game)
        record (bool): whether to record the steps of the test
        record_path (str): the location to save the recording to (if record=True)
    """

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
    Agent that will take a random action on each timestep in a vectorized environment.

    Args:
        env (VecEnv): the vectorized environment
    """

    env.reset()
    while True:
        action = env.action_space.sample()
        env.step(np.tile(action, (env.num_envs, 1)))
        env.render()


def main():
    model = PPO.load('./models/PPO_YoshiIsland2')
    test_model(model, 'YoshiIsland2')


if __name__ == '__main__':
    main()
