import numpy as np
from marlenv.goldmine.basic import Goldmine

class GoldmineRV(Goldmine):
    """
    RV stands for relative view
    """
    def __init__(self, agent_num, view_range):
        self.view_range = view_range
        super().__init__(agent_num)

    def _get_observation(self):
        obs = super()._get_observation()
        mask = self._get_rv_mask()
        for i in range(self.agent_num):
            obs[i, :, :, 1] *= mask[i]
        return obs

    def _get_rv_mask(self):
        mask = np.zeros((self.agent_num, self.height, self.width), dtype=np.float32)
        for i, (y, x) in enumerate(self.agent_pos):
            sy = max(0, y - self.view_range)
            sx = max(0, x - self.view_range)
            ey = min(self.height, y + self.view_range + 1)
            ex = min(self.width, x + self.view_range + 1)
            mask[i, sy:ey, sx:ex] = 1.0
        return mask
