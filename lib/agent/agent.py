class Agent:
    def __init__(self, action_space, observation_space, memory_size):
        self.memory = None
        self.action_space = None
        self.observation_space = None

    def get_action(self, obs):
        raise NotImplementedError

    def set_env(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def train(self, memory):
        pass
