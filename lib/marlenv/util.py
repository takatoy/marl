import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

class GoldmineRecorder:
    def __init__(self, epoch, path, agent_num):
        path += '/epoch{:03d}'.format(epoch)
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

class Debugger:
    cnt = 0

    def draw_observation(observation):
        for i in range(3):
            im = Image.new('RGB', (500, 500), (255, 255, 255))
            draw = ImageDraw.Draw(im)
            for y in range(20):
                for x in range(20):
                    if observation[y, x, i] > 0.005:
                        draw.rectangle((y * 25, x * 25, (y + 1) * 25, (x + 1) * 25), fill=(0, 0, 0))
            im.save('debug/{}_{}.png'.format(Debugger.cnt, i))
        Debugger.cnt += 1

    def draw_state(state):
        im = Image.new('RGB', (500, 500), (255, 255, 255))
        draw = ImageDraw.Draw(im)
        for y in range(20):
            for x in range(20):
                if state[y, x] == 1:
                    draw.rectangle((y * 25, x * 25, (y + 1) * 25, (x + 1) * 25), fill=(255, 0, 0))
                if state[y, x] == 2:
                    draw.rectangle((y * 25, x * 25, (y + 1) * 25, (x + 1) * 25), fill=(0, 0, 255))
        im.save('debug/{:05d}.png'.format(Debugger.cnt))
        Debugger.cnt += 1
        if Debugger.cnt > 200:
            import sys; sys.exit()
