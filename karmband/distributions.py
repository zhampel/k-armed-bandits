from __future__ import print_function

try:
    import numpy as np

    from scipy.stats import norm, binom, bernoulli

except ImportError as e:
    print(e)
    raise ImportError


class Distribution():
    def __init__(self, name='', params=[], verbose=True):
        self.name = name
        self.params = params
        self.verbose = verbose

    def set_params(self, params=[]):
        self.params = params
        if self.verbose:
            print("Setting params of {} dist:".format(self.name))
            self.print_params()

    def get_params(self):
        return self.params

    def print_params(self):
        try:
            self.param_names
        except NameError:
            self.param_names = ['%f'%i for i in self.params]
        for name, param in zip(self.param_names, self.params):
            print('\t{} {}'.format(name, param))

    def pdf(self, x=None):
        return self.dist.pdf(x)

    def rvs(self, size=1):
        return self.dist.rvs(size=size)


class Norm(Distribution):
    def __init__(self, params=[0, 1]):
        super(Norm, self).__init__()
        self.name = 'Gaussian'
        self.params = params
        self.param_names = ['mu', 'sigma']
        self.dist = norm

    def pdf(self, x=None):
        mu = self.params[0]
        sigma = self.params[1]
        return self.dist.pdf(x, loc=mu, scale=sigma)

    def rvs(self, size=1):
        mu = self.params[0]
        sigma = self.params[1]
        return self.dist.rvs(size=size)*sigma + mu


class Binom(Distribution):
    def __init__(self, params=[1, 0.5]):
        super(Norm, self).__init__()
        self.name = 'Binomial'
        self.params = params
        self.param_names = ['n', 'p']
        self.dist = norm

    def pdf(self, x=None):
        n = self.params[0]
        p = self.params[1]
        return self.dist.pmf(x, n, p)

    def rvs(self, size=1):
        n = self.params[0]
        p = self.params[1]
        return self.dist.rvs(n, p, size=size)


DISTRIBUTION_FN_DICT = {
                        'gauss' : Norm(),
                        #'binom' : binom,
                        #'bernoulli' : bernoulli,
                       }


distribution_list = DISTRIBUTION_FN_DICT.keys()


def get_distribution(distribution_name='gauss', params=[0, 1]):
    """
    Convenience function for retrieving
    allowed probability distribution function.
    Parameters
    ----------
    name : {'gauss',...}
          Name of pdf
    Returns
    -------
    fn : function
         SciPy PDF
    """
    if distribution_name in DISTRIBUTION_FN_DICT:
        fn = DISTRIBUTION_FN_DICT[distribution_name]#(params=params)
        fn.set_params(params=params)
        return fn
    else:
        raise ValueError('Invalid PDF name, {}, entered. Must be '
                         'in {}'.format(distribution_name, DISTRIBUTION_FN_DICT.keys()))




#class ():
#    """
#    Class for representing the 
#    details of a single Bandit.
#    """
#    def __init__(self, dist_func=None):
#        super(Bandit, self).__init__()
#        self.dist_func = dist_func
#        self.num_chosen = 0
#        self.cum_reward = 0.
#        self.step_reward = []
#
#    def get_reward(self, time_step=0):
#        reward = self.dist_func().rvs(size=1)
#        self.step_reward.append([time_step, reward])
#        return reward
