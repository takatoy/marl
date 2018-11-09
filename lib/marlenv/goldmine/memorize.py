import numpy as np
from marlenv.goldmine.relative import GoldmineRV

class GoldmineMV(GoldmineRV):
    """
    MV stands for memorized view
    """
    def __init__(self, agent_num, view_range, mem_period, mem_range):
        self.mem_period = mem_period
        self.mem_range = mem_range
        super().__init__(agent_num, view_range)

        self.prev_task_state = np.zeros((self.mem_period, self.height, self.width))

    def step(self, action):
        obs, reward, done, info = super().__init__(action)
        self.prev_task_state = np.roll(self.prev_task_state, 1, axis=0)
        self.prev_task_state[0] = self.task_state
        return obs, reward, done, info

    def _get_observation(self):
        obs = super()._get_observation()
