import os
import numpy as np
import torch as th
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


class CustomCallback(BaseCallback):
    """
    A custom callback that dervies from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: no output 1: info 2: debug
    """
    def __init__(self, verbose: int = 0):
        super(CustomCallback, self).__init__(verbose=verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class SaveBestCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq, log_dir, verbose=0):
        super(SaveBestCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model at {} timesteps".format(x[-1]))
                        print("Saving new best model to {}.zip".format(self.save_path))
                    self.model.save(self.save_path)
        return True


class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)


class ProgressBarManager(object):
    def __init__(self, total_timesteps):
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def __enter__(self):
        self.pbar = tqdm(total=self.total_timesteps)
        return ProgressBarCallback(self.pbar)

    def __exit__(self, type, value, traceback):
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()
    

class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.
    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """
    def __init__(self, eval_env, n_eval_episodes, eval_freq):
        super(EvalCallback, self).__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # TODO evaluate the agent:
            # you need to do self.n_eval_episodes loop using self.eval_env
            # hint: you can use self.model.predict(obs, deterministic=True)
            #save the agent if needed and update self.best_mean_reward
            print('Best mean reward: {:.2f}')
        return True

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq, n_eval_episodes=1, deterministic=True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self):
        if self.n_calls % self._render_freq == 0:
            screens = []
            evaluate_policy(
                self.model,
                self._eval_env,
                callback=lambda l, g: screens.append(self._eval_env.render(mode='rgb_array').transpose(2, 0, 1)),
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic
            )
            self.logger.record(
                'trajectory/video',
                Video(th.ByteTensor([screens]), fps=40),
                exclude=('stdout', 'log', 'json', 'csv')
            )
        return True