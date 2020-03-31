from __future__ import print_function

try:
    import numpy as np

    from scipy.stats import norm, binom, bernoulli

except ImportError as e:
    print(e)
    raise ImportError


class Distribution():
    """
    Generic class for probability distribution
    functions & defining assoc. methods for use.
    """
    def __init__(self, name='', params=[], verbose=True):
        super(Distribution, self).__init__()
        self.name = name
        self.params = params
        self.verbose = verbose

        self.check_params()

    def check_params(self):
        assert_message = \
        "\tNumber of params {} to {} dist not equal to required params {}".\
        format(len(self.params), self.name, self.n_params)
        
        assert self.n_params == len(self.params), assert_message

        try:
            self.param_names
        except NameError:
            self.param_names = ['%f'%i for i in self.params]

        if self.verbose:
            print("Setting params of {} dist:".format(self.name))
            self.print_params()

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
            print('\t{}\t{}'.format(name, param))

    def pdf(self, x=None):
        return self.dist.pdf(x)

    def rvs(self, size=1):
        return self.dist.rvs(size=size)


class Norm(Distribution):
    """
    Normal distribution class.
    """
    def __init__(self, params=[0, 1], verbose=True):
        self.name = 'Gaussian'
        self.n_params = 2
        self.params = params
        self.param_names = ['mu', 'sigma']
        self.dist = norm
        super().__init__(self.name, self.params, verbose)

    def pdf(self, x=None):
        mu = self.params[0]
        sigma = self.params[1]
        return self.dist.pdf(x, loc=mu, scale=sigma)

    def rvs(self, size=1):
        mu = self.params[0]
        sigma = self.params[1]
        return self.dist.rvs(size=size)*sigma + mu


class Binom(Distribution):
    """
    Binomial distribution class.
    """
    def __init__(self, params=[1, 0.5], verbose=True):
        super(Binom, self).__init__()
        self.name = 'Binomial'
        self.params = params
        self.param_names = ['n', 'p']
        self.dist = binom
        super().__init__(self.name, self.params, verbose)

    def pdf(self, x=None):
        n = self.params[0]
        p = self.params[1]
        return self.dist.pmf(x, n, p)

    def rvs(self, size=1):
        n = self.params[0]
        p = self.params[1]
        return self.dist.rvs(n, p, size=size)


# Dictionary for instantiating pdfs from available distributions
DISTRIBUTION_FN_DICT = {
                        'gauss' : Norm,
                        'binom' : Binom,
                        #'bernoulli' : Bernoulli,
                       }


distribution_list = DISTRIBUTION_FN_DICT.keys()


def get_distribution(distribution_name='gauss', params=[0, 1]):
    """
    Convenience function for retrieving
    allowed probability distribution functions.
    Parameters
    ----------
    distribution_name : string {'gauss',...}
        Name of pdf
    params : list
        List of parameters defining pdf
    Returns
    -------
    fn : function
         SciPy PDF
    """
    if distribution_name in DISTRIBUTION_FN_DICT:
        fn = DISTRIBUTION_FN_DICT[distribution_name](params=params)
        return fn
    else:
        raise ValueError('Invalid PDF name, {}, entered. Must be '
                         'in {}'.format(distribution_name, DISTRIBUTION_FN_DICT.keys()))


