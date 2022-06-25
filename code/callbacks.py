import os
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class SaveCheckpoint(BaseCallback):
    def __init__(self, save_freq, name_prefix, model_path, record_path, verbose=0):
        super(SaveCheckpoint, self).__init__(verbose=verbose)
        self.save_freq = save_freq
        self.name_prefix = name_prefix
        self.model_path = model_path
        self.record_path = record_path
        self.record_next_ep = False
        self.recording = False

    def _init_callback(self):
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)
        if self.record_path is not None:
            os.makedirs(self.record_path, exist_ok=True)

    def _on_step(self):
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.model_path,
                f'{self.name_prefix}_{self.num_timesteps}_steps'
            )
            if self.verbose > 0:
                print(f'\nsaving model to {path}')
            self.model.save(path)
            self.record_next_ep = True
        if self.recording:
            if self.locals['dones'][0]:
                if self.verbose > 0:
                    print(f'\nstop recording at {self.num_timesteps} steps')
                env = self.model.get_env().envs[0]
                env.stop_record()
                self.recording = False
        elif self.record_next_ep:
            if self.locals['dones'][0]:
                if self.verbose > 0:
                    print(f'\nstart recording at {self.num_timesteps} steps')
                env = self.model.get_env().envs[0]
                env.record_movie(os.path.join(
                    self.record_path,
                    f'{self.name_prefix}_{self.num_timesteps}.bk2'
                ))
                self.recording = True
                self.record_next_ep = False
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, prog_bar):
        super(ProgressBarCallback, self).__init__()
        self._prog_bar = prog_bar

    def _on_step(self):
        self._prog_bar.n = self.num_timesteps
        self._prog_bar.update(0)


class ProgressBar:
    def __init__(self, initial_timesteps, total_timesteps):
        self.prog_bar = None
        self.initial_timesteps = initial_timesteps
        self.total_timesteps = total_timesteps
    
    def __enter__(self):
        self.prog_bar = tqdm(initial=self.initial_timesteps, total=self.total_timesteps)
        return ProgressBarCallback(self.prog_bar)

    def __exit__(self, type, value, traceback):
        self.prog_bar.n = self.total_timesteps
        self.prog_bar.update(0)
        self.prog_bar.close()
