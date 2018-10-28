import os

class Logger:
    def __init__(self, path):
        self.log_fp = open(path, 'w')

    def log(self, epoch, reward, loss):
        self.log_fp.write('epoch {:d}: reward {:.6f}, loss {:.6f}\n'.format(epoch, reward, loss))
