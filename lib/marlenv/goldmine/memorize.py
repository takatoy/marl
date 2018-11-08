import numpy as np
from marlenv.goldmine.relative import GoldmineRV

class GoldmineMV(GoldmineRV):
    """
    MV stands for memorized view
    """
    def _get_observation(self):
        obs = super()._get_observation()
