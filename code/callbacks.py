import os
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, name_prefix, model_path, record_path, verbose=1):
        super(CheckpointCallback, self).__init__(verbose=verbose)
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
            self.record_next_ep = True
            path = os.path.join(self.model_path, f'{self.name_prefix}_{self.num_timesteps}_steps')
            self.model.save(path)
            if self.verbose > 1:
                print(f'saving model to {path}')
        if self.recording:
            env = self.model.get_env().envs[0]
            if env.data.is_done():
                env.stop_record()
                self.recording = False
        if self.record_next_ep:
            env = self.model.get_env().envs[0]
            if env.data.is_done():
                env.auto_record(self.record_path)               
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


#class EvalCallback(BaseCallback):
#    def __init__(self, eval_env, eval_freq, n_eval_episodes, model_path=None, verbose=1):
#        super(EvalCallback, self).__init__(verbose=verbose)
#        self.eval_env = eval_env
#        self.eval_freq = eval_freq
#        self.n_eval_episodes = n_eval_episodes
#        self.model_path = model_path
#        self.best_mean_reward = -np.inf
#
#    def _init_callback(self):
#        if self.model_path is not None:
#            os.makedirs(self.model_path, exist_ok=True)
#
#    def _on_step(self):
#        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
#            self.eval_env.auto_record('./playback/')
#            episode_rewards, episode_lengths = self._evaluate_model()
#            self.eval_env.stop_record()
#            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
#            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
#            if self.verbose > 0:
#                print(f'\nnum_timesteps: {self.num_timesteps}')
#                print(f'mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}')
#                print(f'mean_length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}')
#            if mean_reward > self.best_mean_reward:
#                if self.model_path is not None:
#                    if self.verbose > 0:
#                        print('new best mean reward!, saving model ...')
#                    self.model.save(os.path.join(self.model_path, f'model_{self.num_timesteps}'))
#                self.best_mean_reward = mean_reward
#        return True
#
#    def _evaluate_model(self):
#        episode_rewards = np.zeros(self.n_eval_episodes)
#        episode_lengths = np.zeros(self.n_eval_episodes, dtype='int')
#        for i in range(self.n_eval_episodes):
#            obs = self.eval_env.reset()
#            ep_reward = 0
#            ep_length = 0
#            done = False
#            while not done:
#                action, _ = self.model.predict(obs, deterministic=True)
#                obs, reward, done, _ = self.eval_env.step(action)
#                ep_reward += reward
#                ep_length += 1
#            episode_rewards[i] = ep_reward
#            episode_lengths[i] = ep_length
#        return episode_rewards, episode_lengths
#