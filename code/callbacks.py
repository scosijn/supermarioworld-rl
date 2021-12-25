import os
import numpy as np
import torch as th
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class ProgressBarCallback(BaseCallback):
    def __init__(self, prog_bar):
        super(ProgressBarCallback, self).__init__()
        self._prog_bar = prog_bar

    def _on_step(self):
        self._prog_bar.n = self.num_timesteps
        self._prog_bar.update(0)


class ProgressBar:
    def __init__(self, total_timesteps):
        self.prog_bar = None
        self.total_timesteps = total_timesteps
    
    def __enter__(self):
        self.prog_bar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.prog_bar)

    def __exit__(self, type, value, traceback):
        self.prog_bar.n = self.total_timesteps
        self.prog_bar.update(0)
        self.prog_bar.close()
