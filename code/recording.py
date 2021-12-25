import os
import base64
import gym
from pathlib import Path
from IPython import display as ipythondisplay
from IPython.display import display, HTML
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
#os.system("Xvfb :1 -screen 0 1024x768x24 &")
#os.environ['DISPLAY'] = ':1'


def show_video(prefix='', video_folder='videos/'):
    """
    Taken from https://github.com/eleurent/highway-env
    :param prefix: (str) Filter the video, showing only the only starting with this prefix
    :param video_folder: (str) Path to the folder containing videos
    """
    html = []
    for video in Path(video_folder).glob('{}*.mp4'.format(prefix)):
        video_b64 = base64.b64encode(video.read_bytes())
        html.append('''
        <video alt="{}" autoplay loop controls style="height: 400px;">
        <source src="data:video/mp4;base64,{}" type="video/mp4" />
        </video>
        '''.format(video, video_b64.decode('ascii')))
    print(display(HTML(data='<br>'.join(html))))


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
    """
    :param env_id: (str)
    :param model: (RL model)
    :param video_length: (int)
    :param prefix: (str)
    :param video_folder: (str)
    """
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecVideoRecorder(eval_env,
                                video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0,
                                video_length=video_length,
                                name_prefix=prefix)
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _ = eval_env.step(action)
    eval_env.close()
