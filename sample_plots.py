import pandas as pd
import numpy as np
from scipy.io import loadmat
from nmf_gpp import link_exp_to_gauss, link_rectgauss
import matplotlib.pyplot as plt

nr_D = 800
nr_H = 100
M = 2


sigma_N = 5
lamb = 1
sigmas = 1
s = 1
burn = 1000
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

chain = pd.read_csv("small01/chain-0.csv")
D_trace_tmp = []
H_trace_tmp = []

for i in range(nr_chains):
    D_trace_tmp.append(chain[D_trace_cols][burn:].values)
    H_trace_tmp.append(chain[H_trace_cols][burn:].values)

D_trace_ = D_trace_tmp[0].copy()
H_trace_ = H_trace_tmp[0].copy()
for i in range(nr_chains - 1):
    D_trace_ += D_trace_tmp[i + 1]
    H_trace_ += H_trace_tmp[i + 1]

D_trace_ = D_trace_/nr_chains
H_trace_ = H_trace_/nr_chains

# D link_exp_to_gauss
# H link_rectgauss
print(D_trace_.shape)
D_trace = np.zeros(D_trace_.shape)
H_trace = np.zeros(H_trace_.shape)

for i in range(len(D_trace_)):
    D_trace[i, :] = link_exp_to_gauss(D_trace_[i].reshape(M, int(nr_D / M)), pars=[1, 1])[0].flatten()

for i in range(len(H_trace_)):
    H_trace[i, :] = link_rectgauss(H_trace_[i].reshape(M, int(nr_H / M)), pars=[1, 1])[0].flatten()

D_sd = np.std(D_trace, axis=0)
H_sd = np.std(H_trace, axis=0)

# Use link function after {
#D_sd = np.std(D_trace_, axis=0)
#H_sd = np.std(H_trace_, axis=0)
# }



# Just a check
assert D_trace.shape[1] == nr_D
assert H_trace.shape[1] == nr_H

D_mean = np.mean(D_trace, axis=0).reshape(M, int(nr_D / M))
H_mean = np.mean(H_trace, axis=0).reshape(M, int(nr_H / M))

# Use link function after {
#D_mean = np.mean(D_trace_, axis=0).reshape(M, int(nr_D / M))
#H_mean = np.mean(H_trace_, axis=0).reshape(M, int(nr_H / M))
# }

D_sd = D_sd.reshape(M, int(nr_D / M))
H_sd = H_sd.reshape(M, int(nr_H / M))

# %%
# Use link function after std and mean {
plt.figure()
plt.fill_between(np.arange(len(H_mean[0,:])), link_rectgauss((H_mean[0,:] - 2*H_sd[0,:]).reshape(1,50),pars=[1,1])[0].flatten(), link_rectgauss((H_mean[0,:] + 2*H_sd[0,:]).reshape(1,50),pars=[1,1])[0].flatten(),color='orange', alpha=.5)
plt.plot(link_rectgauss((H_mean[0,:]).reshape(1,50),pars=[1,1])[0].flatten(), color='orange')
plt.fill_between(np.arange(len(H_mean[1,:])), link_rectgauss((H_mean[1,:] - 2*H_sd[1,:]).reshape(1,50),pars=[1,1])[0].flatten(), link_rectgauss((H_mean[1,:] - 2*H_sd[1,:]).reshape(1,50),pars=[1,1])[0].flatten(),color='blue', alpha=.5)
#plt.plot(link_rectgauss((H_mean[1,:]).reshape(1,50),pars=[1,1])[0].flatten(), color='blue')
plt.plot(H_mean[1, :], color='blue')
spectra = a.T @ vp / 11
plt.plot(spectra[0,:], color='green')
plt.show()
# }


plt.figure()
plt.fill_between(np.arange(len(H_mean[0,:])), H_mean[0,:] - 2*H_sd[0,:], H_mean[0,:] + 2*H_sd[0,:],color='orange', alpha=.5)
plt.plot(H_mean[0, :], color='orange')
plt.fill_between(np.arange(len(H_mean[1,:])), H_mean[1,:] - 2*H_sd[1,:], H_mean[1,:] + 2*H_sd[1,:],color='blue', alpha=.5)
plt.plot(H_mean[1, :], color='blue')
spectra = a.T @ vp / 11
plt.plot(spectra[0,:], color='green')
plt.show()

"""
plt.figure()
plt.fill_between(np.arange(len(D_mean[0,:])), D_mean[0,:] - 2*D_sd[0,:], D_mean[0,:] + 2*D_sd[0,:],color='orange', alpha=.5)
plt.plot(D_mean[0, :], color='orange')
plt.fill_between(np.arange(len(D_mean[1,:])), D_mean[1,:] - 2*D_sd[1,:], D_mean[1,:] + 2*D_sd[1,:],color='blue', alpha=.5)
plt.plot(D_mean[1, :], color='blue')

plt.show()
"""

import pandas as pd
import numpy as np
from scipy.io import loadmat
from nmf_gpp import link_exp_to_gauss, link_rectgauss
import matplotlib.pyplot as plt

nr_D = 800
nr_H = 100
M = 2
burn = 200

lsd_log = []
lsh_log = []
nr_chains = 3

for i in range(nr_chains):
    chain = pd.read_csv("./trace_hej/chain-{}.csv".format(0))
    #chain = pd.read_csv("trace_hej/chain-i.csv".format(i))
    if not i:
        lsd_log = chain["lsd_log____0"].values[burn:]
        lsh_log = chain["lsh_log____0"].values[burn:]
    else:
        lsd_log = np.concatenate((lsd_log, chain["lsd_log____0"].values[burn:]), axis=0)
        lsh_log = np.concatenate((lsh_log, chain["lsh_log____0"].values[burn:]), axis=0)

plt.hist(lsh_log, bins=50)
#plt.savefig("hist_d.png")
plt.show()

plt.hist(lsd_log, bins=50)
plt.savefig("hist_d.png")
plt.show()
