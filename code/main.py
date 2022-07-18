import retro
from wrappers import wrap_env
from callbacks import ProgressBar, SaveCheckpoint
from recording import play_recording, play_all_recordings, recording_to_video
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
    env = wrap_env(env)
    env = Monitor(env)
    check_env(env)
    if verbose > 0:
        print('gamename: {}'.format(env.gamename))
        print('statename: {}'.format(env.statename))
        print('buttons: {}'.format(env.buttons))
        print('action_space: {}'.format(env.action_space))
        print('observation_space: {}'.format(env.observation_space))
        print('reward_range: {}'.format(env.reward_range))
    return env


#def PPO_model(env, log='./tensorboard/'):
#    model = PPO(policy='CnnPolicy',
#                env=env,
#                learning_rate=lambda f: f * 2.5e-4,
#                n_steps=128,
#                batch_size=4,
#                n_epochs=4,
#                ent_coef=.01,
#                clip_range=0.1,
#                gamma=0.99,
#                gae_lambda=0.95,
#                tensorboard_log=log)
#    return model


def PPO_model(env, log='./tensorboard/'):
    model = PPO(policy='CnnPolicy',
                env=env,
                learning_rate=lambda f: f * 2.5e-4,
                n_steps=128,
                n_epochs=4,
                batch_size=32,
                clip_range=lambda f: f * 0.1,
                ent_coef=0.01,
                tensorboard_log=log)
    return model


def DQN_model(env, log='./tensorboard/'):
    model = DQN(policy='CnnPolicy',
                env=env,
                buffer_size=250_000,
                learning_starts=0,
                batch_size=4,
                tensorboard_log=log)
    return model


def train_model(model, total_timesteps, save_freq, name_prefix='model', verbose=0):
    checkpoint_callback = SaveCheckpoint(
        save_freq=save_freq,
        name_prefix=name_prefix,
        model_path='./models/',
        record_path='./recordings/',
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


def grid_search():
    n_steps_arr = [(128, 'steps128')]
    batch_size_arr = [(32, 'batch32'), (128, 'batch128')]
    clip_range_arr = [(0.2, 'const02')] #[(lambda f: f * 0.1, 'lin01'), (0.1, 'const01'), (0.2, 'const02')]
    for steps_value, steps_name in n_steps_arr:
        for batch_value, batch_name in batch_size_arr:
            for clip_value, clip_name in clip_range_arr:
                model_name = steps_name + '_' + batch_name + '_' + clip_name
                env = make_retro_env('YoshiIsland2')
                model = PPO(policy='CnnPolicy',
                            env=env,
                            learning_rate=lambda f: f * 2.5e-4,
                            n_steps=steps_value,
                            batch_size=batch_value,
                            n_epochs=4,
                            clip_range=clip_value,
                            ent_coef=0.01,
                            seed=123,
                            tensorboard_log='./tensorboard/')
                train_model(model,
                            total_timesteps=500_000,
                            save_freq=0,
                            name_prefix=model_name)
                model.save(model_name)
                env.close()


def main():
    env = make_retro_env('YoshiIsland2')


if __name__ == '__main__':
    main()
