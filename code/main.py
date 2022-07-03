import os
import retro
import time
from wrappers import wrap_env
from callbacks import ProgressBar, SaveCheckpoint
from recording import play_recording, play_all_recordings
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


def make_retro_env(state, verbose=0):
    env = retro.RetroEnv(
        game='SuperMarioWorld-Snes',
        state=state,
        info='./data/data.json',
        scenario='./data/scenario.json'
    )
    env = Monitor(wrap_env(env))
    check_env(env)
    if verbose > 0:
        print('gamename: {}'.format(env.gamename))
        print('statename: {}'.format(env.statename))
        print('buttons: {}'.format(env.buttons))
        print('action_space: {}'.format(env.action_space))
        print('observation_space: {}'.format(env.observation_space))
        print('reward_range: {}'.format(env.reward_range))
    return env


def PPO_model(env):
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
    return model


def DQN_model(env):
    model = DQN(policy='CnnPolicy',
                env=env,
                buffer_size=250_000,
                learning_starts=0,
                batch_size=4,
                tensorboard_log='./tensorboard/')
    return model


def train_model(model, total_timesteps, save_freq, name_prefix='model', verbose=0):
    checkpoint_callback = SaveCheckpoint(
        save_freq=save_freq,
        name_prefix=name_prefix,
        model_path='./models/',
        record_path='./playback/',
        verbose=verbose
    )
    with ProgressBar(
        model.num_timesteps,
        model.num_timesteps + total_timesteps
    ) as progress_callback:
        model.learn(
            total_timesteps,
            reset_num_timesteps=False,
            callback=[checkpoint_callback, progress_callback]
        )


def test_model(model, env):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
    env.render(close=True)
    env.close()


def random_agent(env, infinite=False):
    """
    Agent that will take a random action on each timestep.

    Args:
        env (Gym Environment): the environment
        infinite (bool): end after a single episode if false
    """
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done and infinite:
            env.reset()
            done = False
    env.render(close=True)
    env.close()


def main():
    env = make_retro_env('YoshiIsland2')
    model = PPO_model(env)
    train_model(model,
                total_timesteps=25_000,
                save_freq=5_000,
                name_prefix='mario_ppo')

    #model = PPO_model(env)
    #train_model(model,
    #            total_timesteps=1_000_000,
    #            save_freq=100_000,
    #            name_prefix='mario_ppo')


if __name__ == '__main__':
    main()
