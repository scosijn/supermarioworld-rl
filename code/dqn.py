import gym
import retro
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from discretizer import MarioWorldDiscretizer


def dqn(state_shape, action_shape):
    init = HeUniform()
    model = Sequential()
    model.add(Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
    model.add(Dense(12, activation='relu', kernel_initializer=init))
    model.add(Dense(action_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


def train(replay_memory, model, target_model):
    MIN_REPLAY_SIZE = 1000
    BATCH_SIZE = 500
    LEARNING_RATE = 0.7
    DISCOUNT_FACTOR = 0.618
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return
    mini_batch = random.sample(replay_memory, BATCH_SIZE)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)
    X = []
    y = []
    for index, (observation, action, reward, _, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + DISCOUNT_FACTOR * np.max(future_qs_list[index])
        else:
            max_future_q = reward
        current_qs = current_qs_list[index]
        current_qs[action] = (1 - LEARNING_RATE) * current_qs[action] + LEARNING_RATE * max_future_q
        X.append(observation)
        y.append(current_qs)
    model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, shuffle=True)


def train_model(episodes):
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    DECAY = 0.01
    train_episodes = episodes
    epsilon = 1
    env = MarioWorldDiscretizer(retro.make(game='SuperMarioWorld-Snes'))
    input_dim = env.observation_space.shape
    output_dim = env.action_space.n
    model = dqn(input_dim, output_dim)
    target_model = dqn(input_dim, output_dim)
    target_model.set_weights(model.get_weights())
    replay_memory = deque(maxlen=50_000)
    steps = []
    steps_to_update_target_model = 0
    for episode in range(train_episodes):
        n_steps_episode = 0
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            random_number = np.random.rand()
            if random_number <= epsilon:
                action = env.action_space.sample()
            else:
                observation_reshaped = observation.reshape([1, observation.shape[0]])
                predicted = model.predict(observation_reshaped).flatten()
                action = np.argmax(predicted)
            new_observation, reward, done, info = env.step(action)
            total_training_rewards += reward
            replay_memory.append([observation, action, reward, new_observation, done])
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model)
            observation = new_observation
            n_steps_episode += 1
            if done:
                print('{} Total training rewards: {} after n steps = {}'.format(episode, total_training_rewards, n_steps_episode))
                steps.append(n_steps_episode)
                if steps_to_update_target_model >= 100:
                    print('Copying main network weights to the target network weights')
                    target_model.set_weights(model.get_weights())
                    steps_to_update_target_model = 0
                break
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY * episode)
    env.close()
    model.save('SMW_deepQ.h5')


def plot_res(values, title=''):
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(195, c='red',ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x,p(x),"--", label='trend')
    except:
        print('')
    ax[1].hist(values[-50:])
    ax[1].axvline(200, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 50 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    train_model(100)
    #env = gym.make('CartPole-v1')
    #model = load_model('cartpole_deepQ.h5', compile=False)
    #epochs, rewards = 0, 0
    #obs = env.reset()
    #done = False
    #while not done:
    #    env.render()
    #    obs_reshaped = obs.reshape([1, obs.shape[0]])
    #    pred = model.predict(obs_reshaped).flatten()
    #    action = np.argmax(pred)
    #    state, reward, done, info = env.step(action)
    #    rewards += reward
    #    epochs += 1 
    #env.close()
    #print(epochs)