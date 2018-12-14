import numpy as np
from marlenv.goldmine.relative import GoldmineRV

class GoldmineHRV(GoldmineRV):
    """
    HRV stands for historic relative view.
    Tasks remain in the observation for h steps, even after out of view range.
    """
    def __init__(self, agent_num, task_num, view_range, mem_period):
        self.mem_period = mem_period
        super().__init__(agent_num, task_num, view_range)
        self.mem = np.zeros((self.agent_num, self.height, self.width))

    def reset(self):
        self.mem = np.zeros((self.agent_num, self.height, self.width))
        return super().reset()

    def _get_observation(self):
        obs = super()._get_observation()
        mask = self._get_rv_mask()
        self.mem = np.where(self.mem > self.mem_period, 0, self.mem)  # remove expired task
        self.mem = np.where(self.mem > 0, self.mem + 1, self.mem)     # add 1 time step
        self.mem = np.where(mask, obs[:, :, :, 1], self.mem)          # overwrite current obs
        obs[:, :, :, 1] = self.mem.astype(np.bool)
        return obs
