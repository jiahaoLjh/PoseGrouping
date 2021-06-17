import numpy as np
import os


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return True
    return False


class AverageMeter(object):
    def __init__(self):
        self.tot = 0.0
        self.n = 0

    def add(self, value, n=1):
        self.tot += value * n
        self.n += n

    def avg(self):
        n = self.n
        if n == 0:
            mean = np.nan
        else:
            mean = self.tot / n
        return mean

    def sum(self):
        return self.tot

    def reset(self):
        self.tot = 0.0
        self.n = 0
