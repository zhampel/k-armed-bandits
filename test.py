from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    from karmband.bandit import Bandit
    from karmband.distributions import get_distribution, distribution_list, Norm
    from karmband.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError


def main():
    global args
    parser = argparse.ArgumentParser(description="K-Armed Bandit Testing Script")
    parser.add_argument("-t", "--num_time_steps", dest="num_time_steps", default=100, type=int, help="Number of time steps")
    parser.add_argument("-n", "--num_bandits", dest="num_bandits", default=10, type=int, help="Number of bandits")
    parser.add_argument("-b", "--bandit_dist", dest="bandit_dist", default='gauss', choices=distribution_list,  help="Bandit PDF")
    parser.add_argument("-i", "--init_dist", dest="init_dist", default='gauss', choices=distribution_list,  help="Parameter initialization PDF")
    parser.add_argument("-e", "--epsilon", dest="epsilon", default=0.0, type=float, help="Epsilon value for deviating from greedy")
    args = parser.parse_args()

    num_bandits= args.num_bandits
    bandit_dist = args.bandit_dist
    init_dist = args.init_dist
    epsilon = args.epsilon

    bandit_list = []
    
    init_func = get_distribution(init_dist, [1, 3])

    for i in range(num_bandits):
        init_mu = init_func.rvs(size=1)
        init_sigma = 1
        params = [init_mu[0], init_sigma]
        bandit_func = get_distribution(bandit_dist, params)
        bandit = Bandit(samp_func=bandit_func)
        bandit_list.append(bandit)

    dist_name = 'gauss'
    params = [0, 2]
    samp_func = get_distribution(dist_name, params)
    #x = np.linspace(-10, 10, 100)
    #print(samp_func.pdf(x))
    #print(samp_func.rvs(size=1))

    #norm_func = Norm(params=params)
    #print(norm_func.pdf(x))
    #print(norm_func.rvs(size=1))

    bandit1 = Bandit(samp_func=samp_func)
    for i in range(10):
        r1 = bandit1.get_reward()
        print("Reward at step %i: %f"%(i, r1))


if __name__ == "__main__":
    main()
