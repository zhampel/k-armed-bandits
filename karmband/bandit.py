from __future__ import print_function

try:
    import numpy as np
    from .distributions import get_distribution
    
except ImportError as e:
    print(e)
    raise ImportError


class Bandit():
    """
    Class for representing the 
    details of a single Bandit.
    """
    def __init__(self, samp_func=None, samp_params=[]):
        super(Bandit, self).__init__()
        self.samp_params = samp_func.get_params()
        self.samp_func = samp_func
        self.num_chosen = 0
        self.cum_reward = 0.
        self.step_reward = []

    def get_reward(self, time_step=0):
        reward = self.samp_func.rvs(size=1)
        self.step_reward.append([time_step, reward])
        return reward
