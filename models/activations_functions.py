import os
import sys
curr_dir = os.getcwd()+'\models'
sys.path.append(curr_dir)
import numpy as np


class default_activation:
    def __call__(self, x):
        return self.activation(x)

    @staticmethod
    def activation(x):
        return 2 / (1 + np.exp(-x)) - 1

    @staticmethod
    def derivative(hx):
        return ((1 + hx) * (1 - hx)) * 0.5

