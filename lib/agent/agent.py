class Agent:
    def __init__(self, action_space, observation_space, memory_size):
        self.memory = None
        self.action_space = None
        self.observation_space = None

    def get_action(self, obs):
        raise NotImplementedError

    def train(self, memory):
        pass
