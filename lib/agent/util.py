from collections import deque
import numpy as np

class Memory:
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, obs, action, reward, nobs):
        self.buffer.append([obs, action, reward, nobs])

    def get(self, i):
        return np.array(self.buffer[i][0]), \
               np.array(self.buffer[i][1]), \
               np.array(self.buffer[i][2]), \
               np.array(self.buffer[i][3])

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return np.array([self.buffer[i][0] for i in idx]), \
               np.array([self.buffer[i][1] for i in idx]), \
               np.array([self.buffer[i][2] for i in idx]), \
               np.array([self.buffer[i][3] for i in idx])

class EpsilonExponentialDecay:
    def __init__(self, init, rate):
        self.init = init
        self.rate = rate

    def get(self, e):
        return self.init * (self.rate ** e)

class EpsilonLinearDecay:
    def __init__(self, init, end, epochs):
        self.init = init
        self.end = end
        self.epochs = epochs

    def get(self, e):
        return max(self.init - (self.init - self.end) / self.epochs * e, self.end)
