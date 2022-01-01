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
    print_env_info(env)
    return env


def print_env_info(env):
    print('gamename: {}'.format(env.gamename))
    print('statename: {}'.format(env.statename))
    print('buttons: {}'.format(env.buttons))
    print('action_space: {}'.format(env.action_space))
    print('observation_space: {}'.format(env.observation_space.shape))
    print('reward_range: {}'.format(env.reward_range))


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


def test_model(env, model_name):
    model = PPO.load(f'./models/{model_name}')
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()


def main():
    env = create_env('YoshiIsland2')
    model = PPO_model(env)
    train_model(model,
                total_timesteps=1_000_000,
                save_freq=100_000,
                name_prefix='mario_ppo')


if __name__ == '__main__':
    main()
