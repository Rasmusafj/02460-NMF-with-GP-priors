from scipy.io import loadmat
import numpy as np
from core import nmf_gpp_hmc

np.random.seed(125)

mat = loadmat('../data/sampling.mat')
X = mat["X"]

K, L = X.shape
M = 2

trace = nmf_gpp_hmc(X, M, numSamples=20, dimD=2, dimH=1, numChains=3,
        db_name='chains')