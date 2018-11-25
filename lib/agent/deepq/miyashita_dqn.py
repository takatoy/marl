from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, concatenate, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from agent.deepq.simple_dqn import SimpleDQN

class MiyashitaDQN(SimpleDQN):
    def _get_model(self):
        obs_in = Input(shape=self.observation_space, dtype='float32')

        # DQN paper-like network
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
