import gym

class MarioWorldMonitor(gym.Wrapper):
    """
    :param env: gym environment that will be wrapped
    """
    def __init__(self, env):
        super(MarioWorldMonitor, self).__init__(env)
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []

    def reset(self):
        """
        Reset the environment
        :return: first observation of the environment
        """
        if not self.needs_reset:
            raise RuntimeError('Tried to reset environment that is not done')
        self.rewards = []
        self.needs_reset = False
        return self.env.reset()

    def step(self, action):
        """
        :param action: action taken by the agent
        :return: observation, reward, done, info
        """
        if self.needs_reset:
            raise RuntimeError('Tried to step environment that needs reset')
        obs, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            curr_ep_reward = sum(self.rewards)
            curr_ep_length = len(self.rewards)
            curr_ep_info = {'r': curr_ep_reward, 'l': curr_ep_length}
            self.episode_rewards.append(curr_ep_reward)
            self.episode_lengths.append(curr_ep_length)
            info['episode'] = curr_ep_info
        return obs, reward, done, info

    def close(self):
        """
        Close the environment
        """
        super(MarioWorldMonitor, self).close()
