import os
import sys
curr_dir = os.getcwd()+'\models'
sys.path.append(curr_dir)

import numpy as np


class normal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, nodes, num_input):
        return np.random.normal(self.mean, self.std, (nodes, num_input))
