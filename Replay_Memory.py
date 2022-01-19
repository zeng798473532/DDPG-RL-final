import numpy as np
import random


class Replay_Memory(object):
    def __init__(self, pool_size, batch_size):
        self.pool = []
        self.max_size = pool_size
        self.index = 0
        self.batch_size = batch_size

    def append(self, state, action, reward, next_state, done):
        """
        将一组参数放入到经验池中
        :return:
        """
        if len(self.pool) != self.max_size:
            self.pool.append((state, action, reward, next_state, done))
            self.index = (self.index + 1) % self.max_size
        else:
            self.pool[self.index] = (state, action, reward, next_state, done)
            self.index = (self.index + 1) % self.max_size

    def sample(self):
        """
        采样：随机从缓存中取出一组值进行学习
        :return: list[(state, action, reward, next_state), (),() ...]
        """
        rand_indexs = random.sample(range(0, len(self.pool)), self.batch_size)
        rand_data = [self.pool[i] for i in rand_indexs]
        return rand_data
