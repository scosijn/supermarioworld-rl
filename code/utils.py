import retro
import gym
import os
import glob
import re
import time
import pandas as pd
import matplotlib.pyplot as plt


def _replay(path, render=False, render_delay=0, video_folder=None):
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players
    )
    env.initial_state = movie.get_state()
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder, lambda *_: True)
    env.reset()
    while movie.step():
        actions = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                actions.append(movie.get_key(i, p))
        env.step(actions)
        if render:
            env.render()
            time.sleep(render_delay)
    if render:
        env.render(close=True)
    env.close()
    movie.close()


def recording_to_video(path, video_folder):
    _replay(path, video_folder=video_folder)


def play_recording(path):
    _replay(path, render=True, render_delay=0.01)


def play_all_recordings(path):
    files = glob.glob(os.path.join(path, '*.bk2'))
    for file_path in sorted(files, key=lambda x:int(re.findall('(\d+)',x)[0])):
        print(f'playing {file_path}')
        play_recording(file_path)
        

def plot_rollout(path, smoothing=0, title='', color='tab:blue'):
    df = pd.read_csv(path)
    plt.plot(df['Step'], df['Value'], alpha=0.2, color=color)
    plt.plot(df['Step'], df['Value'].ewm(alpha=(1 - smoothing)).mean(), color=color)
    plt.xlabel('Timestep')
    plt.title(title)
    plt.show()
