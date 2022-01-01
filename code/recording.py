import retro
import os
import glob
import re
import time


def play_recording(path):
    movie = retro.Movie(path)
    movie.step()
    env = retro.make(
        game=movie.get_game(),
        state=None,
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players
    )
    env.initial_state = movie.get_state()
    env.reset()
    while movie.step():
        actions = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                actions.append(movie.get_key(i, p))
        env.step(actions)
        env.render()
    env.close()
    movie.close()


def play_all_recordings(path):
    files = glob.glob(os.path.join(path, '*.bk2'))
    for file_path in sorted(files, key=lambda x:int(re.findall('(\d+)',x)[0])):
        print(f'playing {file_path}')
        play_recording(file_path)
