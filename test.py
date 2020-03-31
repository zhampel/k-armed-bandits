from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    from karmband.bandit import Bandit
    from karmband.distributions import get_distribution, Norm
    from karmband.plots import plot_train_loss
except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    #parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    #parser.add_argument("-r", "--run_name", dest="run_name", default='karmband', help="Name of training run")
    #parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    #parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    #parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    #args = parser.parse_args()

    dist_name = 'gauss'
    params = [0, 2]
    samp_func = get_distribution(dist_name, params)
    #x = np.linspace(-10, 10, 100)
    #print(samp_func.pdf(x))
    #print(samp_func.rvs(size=1))

    #norm_func = Norm(params=params)
    #print(norm_func.pdf(x))
    #print(norm_func.rvs(size=1))

    bandit1 = Bandit(samp_func=samp_func, samp_params=params)
    for i in range(10):
        r1 = bandit1.get_reward()
        print("Reward at step %i: %f"%(i, r1))


if __name__ == "__main__":
    main()
