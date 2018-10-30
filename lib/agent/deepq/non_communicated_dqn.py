import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, concatenate, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K

from agent.agent import Agent
from agent.util import Memory

class NonCommunicatedDQN(Agent):
    def __init__(self, action_space, observation_space, memory_size,
                 batch_size, learning_rate, gamma, target_update, use_dueling):
        # parameters
        self.observation_space = observation_space
        self.action_space      = action_space
        self.memory_size       = memory_size
        self.batch_size        = batch_size
        self.learning_rate     = learning_rate
        self.gamma             = gamma
        self.target_update     = target_update
        self.use_dueling       = use_dueling

        # variables
        self.train_cnt = 0
        self.memory = Memory(memory_size)

        self.target_network = self.__get_model()
        self.eval_network   = self.__get_model()

    def __get_model(self):
        obs_in = Input(shape=self.observation_space, dtype='float32')

        # DQN paper network
        x = Conv2D(16, 4, 2, padding='same', activation='relu')(obs_in)
        x = Conv2D(32, 2, 1, padding='same', activation='relu')(x)
        x = Conv2D(32, 2, 1, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        # Miyashita network
        # x = Conv2D(8, 2, padding='same')(obs_in)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Conv2D(16, 2, padding='same')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        # x = Flatten()(x)
        # x = Dense(100, activation='relu')(x)

        if self.use_dueling:
            value = Dense(1)(x)
            advantage = Dense(self.action_space, use_bias=False)(x)
            q_vals = Lambda(lambda a: a[0] + (a[1] - K.mean(a[1], keepdims=True)),
                output_shape=(self.action_space,))([value, advantage])
        else:
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
        batch_target = self.__calc_target(batch_obs, batch_action, batch_reward, batch_nobs)
        loss = self.eval_network.train_on_batch(batch_obs, batch_target)

        self.train_cnt += 1
        if self.train_cnt % self.target_update == 0:
            self.__update_target()

        return loss

    def __calc_target(self, batch_obs, batch_action, batch_reward, batch_nobs):
        n = len(batch_action)

        target_q_vals = self.target_network.predict(batch_nobs, batch_size=self.batch_size)
        eval_q_vals = self.eval_network.predict(batch_nobs, batch_size=self.batch_size)
        next_vals = target_q_vals[np.arange(n), np.argmax(eval_q_vals, axis=1)]
        next_vals = batch_reward + self.gamma * next_vals

        targets = self.target_network.predict(batch_obs, batch_size=self.batch_size)
        targets[np.arange(n), batch_action] = next_vals

        return targets

    def __update_target(self):
        self.target_network.set_weights(self.eval_network.get_weights())

    def save(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        self.target_network.save(path + '/target_network_{:06d}'.format(epoch) + '.h5')
        self.eval_network.save(path + '/eval_network_{:06d}'.format(epoch) + '.h5')

    def load(self, path, epoch):
        self.target_network = load_model(path + '/target_network_{:06d}'.format(epoch) + '.h5')
        self.eval_network = load_model(path + '/eval_network_{:06d}'.format(epoch) + '.h5')
