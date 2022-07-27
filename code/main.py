import time
from turtle import pen
import retro
import itertools
import numpy as np
from wrappers import wrap_env, MarioWrapper
from callbacks import ProgressBar, CheckpointCallback
from recording import play_recording, play_all_recordings, recording_to_video
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
            n_skip=4,
            rewards=(-15, 15)
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


#def make_retro_env(state, verbose=0):
#    env = retro.RetroEnv(
#        game='SuperMarioWorld-Snes',
#        state=state,
#        info='./data/data.json',
#        scenario='./data/scenario.json'
#    )
#    env = wrap_env(env)
#    env = Monitor(env)
#    if verbose > 0:
#        print('gamename: {}'.format(env.gamename))
#        print('statename: {}'.format(env.statename))
#        print('buttons: {}'.format(env.buttons))
#        print('action_space: {}'.format(env.action_space))
#        print('observation_space: {}'.format(env.observation_space))
#        print('reward_range: {}'.format(env.reward_range))
#    return env


def PPO_model(env, log='./tensorboard/'):
    model = PPO(policy='CnnPolicy',
                env=env,
                learning_rate=lambda f: f * 1e-4,
                n_steps=1024,
                batch_size=512,
                n_epochs=10,
                clip_range=0.2,
                ent_coef=0.01,
                tensorboard_log=log)
    return model


def train_model(model, total_timesteps, save_freq, name_prefix='model', verbose=0):
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./models/',
        name_prefix=name_prefix,
        verbose=verbose
    )
    with ProgressBar(
        model.num_timesteps,
        model.num_timesteps + total_timesteps
    ) as progress_callback:
        model.learn(
            total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=name_prefix,
            callback=[checkpoint_callback, progress_callback]
        )


def test_model(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
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


#def grid_search():
#    n_steps_arr = [128, 512, 1024]
#    batch_size_arr = [256, 1024]
#    grid = itertools.product(n_steps_arr, batch_size_arr)
#    for n_steps, batch_size in grid:
#        model_name = '_'.join(['n_steps'+str(n_steps), 'batch_size'+str(batch_size)])
#        env = make_retro_env_vec('YoshiIsland2', n_envs=8)
#        model = PPO(policy='CnnPolicy',
#                    env=env,
#                    learning_rate=lambda f: f * 2.5e-4,
#                    n_steps=n_steps,
#                    batch_size=batch_size,
#                    ent_coef=0.01,
#                    seed=42,
#                    tensorboard_log='./tensorboard/')
#        train_model(model,
#                    total_timesteps=500_000,
#                    save_freq=0,
#                    name_prefix=model_name)
#        model.save('./models/' + model_name)
#        env.close()


def main():
    n_envs = 12
    total_timesteps = 40_000_000
    save_freq = (0.25 * total_timesteps) // n_envs
    env = make_retro_env('YoshiIsland2', n_envs=n_envs)
    model = PPO_model(env)
    train_model(model,
                total_timesteps=total_timesteps,
                save_freq=save_freq,
                name_prefix='ppo')
    model.save('./models/ppo_final')
    env.close()


if __name__ == '__main__':
    main()
