import retro
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wrappers import wrap_env
from recording import playback
from discretizer import MarioWorldDiscretizer
from callbacks import ProgressBar, EvalCallback

#A      = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] # spin
#B      = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # jump
#X      = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] # run
#UP     = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
#DOWN   = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
#LEFT   = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
#RIGHT  = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]


def create_env(state):
    env = retro.RetroEnv(
        game='SuperMarioWorld-Snes',
        state=state,
        info='./data/data.json',
        scenario='./data/scenario.json',
    )
    # convert to discrete action space and apply wrappers
    env = MarioWorldDiscretizer(env)
    env = wrap_env(env)
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


def train_model(model, total_timesteps, eval_env, eval_freq, n_eval_episodes, verbose=1):
    eval_callback = EvalCallback(
        eval_env,
        eval_freq,
        n_eval_episodes,
        model_path='./models/',
        verbose=verbose
    )
    model.learn(total_timesteps, callback=eval_callback)
    #with ProgressBar(total_timesteps) as progress_callback:
    #    callback = [progress_callback]
    #    if eval_callback is not None:
    #        callback.append(eval_callback)
    #model.learn(total_timesteps, callback=callback)


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
    GAME = 'Airstriker-Genesis'
    env = retro.RetroEnv(GAME)
    #eval_env.auto_record('./playback/')
    os.makedirs('./playback/', exist_ok=True)
    model = PPO('CnnPolicy', env)
    train_model(model,
                total_timesteps=500,
                eval_env=env,
                eval_freq=250,
                n_eval_episodes=1)


if __name__ == '__main__':
    main()
    #playback('./playback/Airstriker-Genesis-Level1-000004.bk2')
    #playback('./playback/Airstriker-Genesis-Level1-000005.bk2')
