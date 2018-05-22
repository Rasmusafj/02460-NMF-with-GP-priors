import pandas as pd
import numpy as np
from scipy.io import loadmat
from core import link_exp_to_gauss, link_rectgauss
import matplotlib.pyplot as plt

# Dimensions
nr_D = 800
nr_H = 100
M = 2

# Variables to be chosen
burn = 500
nr_chains = 1

mat = loadmat('../data/sampling.mat')
a = mat['a']
vp = mat['vp']

D_trace_cols = []
for i in range(nr_D):
    D_trace_cols.append("d__{0}".format(i))

H_trace_cols = []
for i in range(nr_H):
    H_trace_cols.append("h__{0}".format(i))

for i in range(nr_chains):
    # Get chain
    chain = pd.read_csv("small01/chain-{}.csv".format(i))

    # Get values of trace
    D_trace_ = chain[D_trace_cols][burn:].values
    H_trace_ = chain[H_trace_cols][burn:].values

    # Just a check
    assert D_trace_.shape[1] == nr_D
    assert H_trace_.shape[1] == nr_H

    # Mean and standard deviation
    D_mean = np.mean(D_trace_, axis=0).reshape(M, int(nr_D / M))
    H_mean = np.mean(H_trace_, axis=0).reshape(M, int(nr_H / M))

    D_sd = np.std(D_trace_, axis=0)
    H_sd = np.std(H_trace_, axis=0)

    D_sd = D_sd.reshape(M, int(nr_D / M))
    H_sd = H_sd.reshape(M, int(nr_H / M))

    # Confidence intervals in original domain
    H_li = np.zeros(H_mean.shape)
    H_ui = np.zeros(H_mean.shape)
    H_m = np.zeros(H_mean.shape)

    for i in range(M):
        H_li[i, :] = link_rectgauss(H_mean[i, :] - 2 * H_sd[i, :], pars=[1, 1])[0].flatten()
        H_ui[i, :] = link_rectgauss(H_mean[i, :] + 2 * H_sd[i, :], pars=[1, 1])[0].flatten()
        H_m[i, :] = link_rectgauss(H_mean[i, :], pars=[1, 1])[0].flatten()

    # True spectrum
    spectra = a.T @ vp

    # Plots
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(spectra[0,:], color='green')
    plt.legend(["True Spectrum"], loc = 2)
    plt.subplot(2,1,2)
    plt.fill_between(np.arange(len(H_mean[1,:])), H_li[1,:], H_ui[1,:],color='blue', alpha=.5)
    plt.plot(H_m[1, :], color='blue')
    plt.legend(["Mean","95% Confidence Int."], loc = 2)
    plt.savefig("testiness_simulation.png")
    plt.show()

for i in range(nr_chains):
    chain = pd.read_csv("small01/chain-{}.csv".format(i))
    if not i:
        lsd_log = chain["lsd_log____0"].values[burn:]
        lsh_log = chain["lsh_log____0"].values[burn:]
    else:
        lsd_log = np.concatenate((lsd_log, chain["lsd_log____0"].values[burn:]), axis=0)
        lsh_log = np.concatenate((lsh_log, chain["lsh_log____0"].values[burn:]), axis=0)

    # Histogram of width of covariance for H
    plt.hist(lsh_log, bins=50)
    plt.savefig("hist_h.png")
    plt.show()

    # Histogram of width of covariance for D
    plt.hist(lsd_log, bins=50)
    plt.savefig("hist_d.png")
    plt.show()


    # %%