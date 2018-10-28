import numpy as np

from agent.agent import Agent
from agent.util import Memory

class RandomAgent(Agent):
    def __init__(self, action_space, observation_space, memory_size):
        self.action_space = action_space
        self.memory = Memory(memory_size)

    def get_action(self, obs):
        return np.random.randint(self.action_space, dtype=np.int16)
