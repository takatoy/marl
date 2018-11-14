import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, concatenate, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

from agent.agent import Agent
from agent.util import Memory

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

class MiyashitaDQN(Agent):
    def __init__(self, action_space, observation_space, memory_size,
                 batch_size, learning_rate, gamma, target_update):
        # parameters
        self.observation_space = observation_space
        self.action_space      = action_space
        self.memory_size       = memory_size
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.gamma             = gamma
        self.target_update     = target_update

        # variables
        self.train_cnt = 0
        self.memory = Memory(memory_size)

        self.target_network = self._get_model()
        self.eval_network   = self._get_model()

    def _get_model(self):
        obs_in = Input(shape=self.observation_space, dtype='float32')

        # DQN paper network
        x = Conv2D(8, 2, 1, padding='same')(obs_in)
        x = MaxPooling2D(2, 2)(x)
        x = Conv2D(16, 2, 1, padding='same')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Flatten()(x)
        x = Dense(100, activation='relu')(x)
        q_vals = Dense(self.action_space)(x)

        model = Model(inputs=obs_in, outputs=q_vals)
        optimizer = RMSprop(lr=self.learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model

    def get_action(self, obs, eps):
        if eps < np.random.uniform(0, 1):
            q_vals = self.eval_network.predict(np.expand_dims(obs, axis=0), batch_size=1)
            action = np.argmax(q_vals[0])
        else:
            action = np.random.randint(self.action_space)

        return action

    def train(self):
        batch_obs, batch_action, batch_reward, batch_nobs = self.memory.sample(self.batch_size)
        batch_target = self._calc_target(batch_obs, batch_action, batch_reward, batch_nobs)
        loss = self.eval_network.train_on_batch(batch_obs, batch_target)

        self.train_cnt += 1
        if self.train_cnt % self.target_update == 0:
            self._update_target()

        return loss

    def _calc_target(self, batch_obs, batch_action, batch_reward, batch_nobs):
        n = len(batch_action)

        target_q_vals = self.target_network.predict(batch_nobs, batch_size=self.batch_size)
        eval_q_vals = self.eval_network.predict(batch_nobs, batch_size=self.batch_size)
        next_vals = target_q_vals[np.arange(n), np.argmax(eval_q_vals, axis=1)]
        next_vals = batch_reward + self.gamma * next_vals

        targets = self.target_network.predict(batch_obs, batch_size=self.batch_size)
        targets[np.arange(n), batch_action] = next_vals

        return targets

    def _update_target(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def save(self, path, episode, i):
        if not os.path.exists(path):
            os.makedirs(path)
        self.target_network.save(path + '/target_network_{:06d}'.format(episode, i) + '.h5')
        self.eval_network.save(path + '/eval_network_{:06d}'.format(episode, i) + '.h5')

    def load(self, path, episode, i):
        self.target_network = load_model(path + '/target_network_{:06d}_{:02d}'.format(episode, i) + '.h5')
        self.eval_network = load_model(path + '/eval_network_{:06d}_{:02d}'.format(episode, i) + '.h5')
