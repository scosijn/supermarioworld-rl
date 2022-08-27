import os
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f'{self.name_prefix}_{self.num_timesteps}_steps'
            )
            if self.verbose > 0:
                print(f'\nsaving model to {path}')
            self.model.save(path)
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, progress_bar):
        super().__init__()
        self._progress_bar = progress_bar

    def _on_step(self):
        self._progress_bar.n = self.num_timesteps
        self._progress_bar.update(0)


class ProgressBar:
    def __init__(self, initial_timesteps, total_timesteps):
        self.progress_bar = None
        self.initial_timesteps = initial_timesteps
        self.total_timesteps = total_timesteps
    
    def __enter__(self):
        self.progress_bar = tqdm(initial=self.initial_timesteps, total=self.total_timesteps)
        return ProgressBarCallback(self.progress_bar)

    def __exit__(self, type, value, traceback):
        self.progress_bar.n = self.total_timesteps
        self.progress_bar.update(0)
        self.progress_bar.close()
