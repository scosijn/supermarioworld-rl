import os
import numpy as np
from tqdm.auto import tqdm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EventCallback


class SaveBestCallback(BaseCallback):
    def __init__(
        self,
        eval_freq,
        n_eval_episodes,
        model_path=None,
        record_path=None,
        verbose=1
    ):
        super(SaveBestCallback, self).__init__(verbose=verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.model_path = model_path
        self.record_path = record_path
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        # create folders if needed
        if self.record_path is not None:
            os.makedirs(self.record_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
            os.makedirs(self.model_path, exist_ok=True)

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            return True
        return True


class EvalCallback(EventCallback):
    def __init__(
        self,
        eval_env,
        eval_freq,
        n_eval_episodes=5,
        warn=True,
        verbose=1,
        deterministic=True,
        record_path=None,
        model_path=None,
        log_path=None
    ):
        super(EvalCallback, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.warn = warn
        self.record_path = record_path
        self.model_path = model_path
        if log_path is not None:
            log_path = os.path.join(log_path, 'evaluations')
        self.log_path = log_path
        self.best_mean_reward = -np.inf
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_lengths = []
        self._is_success_buffer = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # reset success rate buffer
            self._is_success_buffer = []
            # record actions taken during evaluation
            if self.record_path is not None:
                self.eval_env.auto_record(self.record_path)
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                warn=self.warn,
                return_episode_rewards=True,
                callback=self._log_success_callback
            )
            # stop recording
            self.eval_env.stop_record()
            # save log file
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_lengths.append(episode_lengths)
                np.savez(self.log_path,
                         timesteps=self.evaluations_timesteps,
                         results=self.evaluations_results,
                         ep_lengths=self.evaluations_lengths)
            # calculate mean and std reward/ep_length over n_eval_episodes
            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            if self.verbose > 0:
                print(f'num_timesteps: {self.num_timesteps}')
                print(f'mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}')
                print(f'mean_length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}')
            # save best model
            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print('New best mean reward!')
                if self.model_path is not None:
                    self.model.save(os.path.join(self.model_path, 'best_model'))
                self.best_mean_reward = mean_reward
                # trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

    def _init_callback(self):
        # create folders if needed
        if self.record_path is not None:
            os.makedirs(self.record_path, exist_ok=True)
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_, globals_):
        info = locals_['info']
        if locals_['done']:
            is_success = info.get('is_success')
            if is_success is not None:
                self._is_success_buffer.append(is_success)

    def update_child_locals(self, locals_):
        if self.callback:
            self.callback.update_locals(locals_)


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
