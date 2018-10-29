import numpy as np
from marlenv.env import Env

class Goldmine(Env):
    """
    Goldmine Environment

    Actions: 0: Move Up, 1: Move Right, 2: Move Down, 3: Move Left
    """
    def __init__(self, agent_num):
        # parameters
        self.height = 20
        self.width = 20
        self.agent_num = agent_num
        self.action_space = 4
        self.observation_space = (self.height, self.width, 3)  # self position, task positions, other agents positions
        self.task_num = 25

        # variables
        self.step_cnt = 0
        self.agent_pos = None
        self.agent_state = None
        self.task_state = None
        self.action = np.array([-1] * self.agent_num, dtype=np.int16)
        self.reward = np.zeros((self.agent_num,), dtype=np.float32)
        self.task_update = []
        self.move_map = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        self.reset()

    def reset(self):
        self.step_cnt = 0
        self.action = np.array([-1] * self.agent_num, dtype=np.int16)
        self.reward = np.zeros((self.agent_num,), dtype=np.float32)

        self.agent_pos = np.zeros((self.agent_num, 2), dtype=np.int16)
        self.agent_state = np.zeros((self.height, self.width), dtype=np.bool)
        self.task_state = np.zeros((self.height, self.width), dtype=np.bool)
        self.task_update = []

        sample = np.random.choice(self.height * self.width, self.agent_num + self.task_num, replace=False)
        for i in range(self.agent_num + self.task_num):
            y = sample[i] // self.width
            x = sample[i] % self.width
            if i < self.agent_num:
                self.agent_pos[i] = [y, x]
                self.agent_state[y, x] = 1
            else:
                self.task_state[y, x] = 1
                self.task_update.append([self.step_cnt, -1, -1, -1, y, x])

        return self._get_observation()

    def step(self, action):
        assert len(action) == self.agent_num
        assert action.dtype == 'int16'

        self.action = action
        self.reward = np.zeros((self.agent_num,), dtype=np.float32)
        done = False # done is always False in this environment

        task_finished = []
        action_order = np.arange(self.agent_num)
        np.random.shuffle(action_order)
        for i in action_order:
            npos = self.agent_pos[i] + self.move_map[action[i]]
            if not npos[0] < 0 and not npos[0] >= self.height and \
               not npos[1] < 0 and not npos[1] >= self.width  and \
               not self.agent_state[npos[0], npos[1]]:
                self.agent_state[self.agent_pos[i][0], self.agent_pos[i][1]] = 0
                self.agent_pos[i] = npos
                self.agent_state[npos[0], npos[1]] = 1

            y, x = self.agent_pos[i]
            if self.task_state[y, x]:
                self.reward[i] = 1.0
                self.task_state[y, x] = 0
                task_finished.append([i, y, x])

        for i, y, x in task_finished:
            ny, nx = self._spawn_task()
            self.task_update.append([self.step_cnt, i, y, x, ny, nx])

        self.step_cnt += 1

        return np.array(self._get_observation()), np.array(self.reward), done, {}

    def render(self, mode='log'):
        ret = None
        if mode == 'log':
            ret = {
                'agent': [[
                    self.step_cnt,         # step number
                    i,                     # agent id
                    self.agent_pos[i][0],  # y coordinate AFTER action taken
                    self.agent_pos[i][1],  # x coordinate AFTER action taken
                    self.action[i],        # action id
                    self.reward[i]         # reward obtained in this step
                ] for i in range(self.agent_num)],
                'task': list(self.task_update)
            }
            self.task_update.clear()
        elif mode == 'state':
            ret = np.zeros((self.height, self.width), dtype=np.int16)  # 0 is empty space
            ret += self.agent_state                                    # 1 is agent
            ret += self.task_state * 2                                 # 2 is task
        return ret

    def close(self):
        pass

    def seed(self, seed=0):
        np.random.seed(seed=seed)

    def _get_observation(self):
        observation = np.zeros((self.agent_num,) + self.observation_space, dtype=np.float32)
        observation[np.arange(self.agent_num), :, :, 1] = self.task_state.astype(np.float32)
        observation[np.arange(self.agent_num), :, :, 2] = self.agent_state.astype(np.float32)

        ys, xs = self.agent_pos[:, 0], self.agent_pos[:, 1]
        observation[np.arange(self.agent_num), ys, xs, 0] = 1.0
        observation[np.arange(self.agent_num), ys, xs, 2] = 0.0

        return observation

    def _spawn_task(self):
        while True:
            n = np.random.randint(self.height * self.width)
            y = n // self.width
            x = n % self.width
            if not self.task_state[y, x] and not self.agent_state[y, x]:
                self.task_state[y, x] = 1
                return [y, x]
