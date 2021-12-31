import retro
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wrappers import wrap_env
from callbacks import ProgressBar, SaveCheckpoint
from recording import play_recording, play_all_recordings


def create_env(state):
    env = retro.RetroEnv(
        game='SuperMarioWorld-Snes',
        state=state,
        info='./data/data.json',
        scenario='./data/scenario.json',
    )
    env = wrap_env(env)
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


def train_model(model, total_timesteps, save_freq, name_prefix='model', verbose=1):
    checkpoint_callback = SaveCheckpoint(
        save_freq=save_freq,
        name_prefix=name_prefix,
        model_path='./models/',
        record_path='./playback/',
        verbose=verbose
    )
    with ProgressBar(total_timesteps) as progress_callback:
        model.learn(total_timesteps, callback=[checkpoint_callback, progress_callback])


def create_PPO_model(env):
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
    env = create_env('YoshiIsland2')
    model = create_PPO_model(env)
    train_model(model, 20000, 10000, 'PPO_Mario')
    print('done')
    return


if __name__ == '__main__':
    main()
