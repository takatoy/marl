import os
import numpy as np

class GoldmineRecorder:
    def __init__(self, epoch, path, agent_num):
        path += '/epoch{:06d}'.format(epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        self.task_fp = open(path + '/task.log', 'w')
        self.agent_fp = []
        for i in range(agent_num):
            name = path + '/agent{:03d}.log'.format(i)
            self.agent_fp.append(open(name, 'w'))

    def record(self, data):
        for d in data['task']:
            self.task_fp.write(','.join(map(str, d)) + '\n') # (step, y, x, ny, nx)
        for i, d in enumerate(data['agent']):
            self.agent_fp[i].write(','.join(map(str, d)) + '\n')

    def close(self):
        self.task_fp.close()
        for f in self.agent_fp:
            f.close()
