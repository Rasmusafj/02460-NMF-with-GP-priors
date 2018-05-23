from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from core import nmf_ls, nmf_gpp_map, link_exp_to_gauss, link_rectgauss, rbf, get_2d_rbf_kernel

file_name1 = '50x50_nw250_noise12_nhs2_k4_21'
mat1 = loadmat('./data/'+file_name1+'.mat')
X1 = mat1["X"]

file_name2 = '50x50_nw250_noise12_nhs2_k4_22'
mat2 = loadmat('./data/'+file_name2+'.mat')
X2 = mat2["X"]

X = X1 + X2

file_name_result = '2_spec_50x50_noise12_nw250_nhs2_k4'

K, L = X.shape
M = 3
beta_H = 10
beta_D = 10
sigma_N = 2
np.random.seed(125)

cov_D_1d = np.zeros((K, K))
for i in range(K):
   for j in range(K):
       cov_D_1d[i, j] = rbf(beta_D, i + 1, j + 1)

dim = int(sqrt(len(X)))
cov_D_2d = get_2d_rbf_kernel(beta_D, (dim,dim))

cov_D_1d = cov_D_1d + 1e-5 * np.eye(K)
cov_D_2d = cov_D_2d + 1e-5 * np.eye(K)

cov_H = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        cov_H[i, j] = rbf(beta_H, i + 1, j + 1)

cov_H = cov_H + 1e-5 * np.eye(L)


print("\nComputing GPP NMF 1d.")
D_1d, H_1d = nmf_gpp_map(X, M, MaxIter=500, covH=cov_H, covD=cov_D_1d, linkH=link_exp_to_gauss, argsH=[np.diag(cov_H), 1],
               linkD=link_rectgauss, argsD=[np.diag(cov_D_1d), 1], sigma_N=sigma_N)

print("\nComputing GPP NMF 2d.")
D_2d, H_2d = nmf_gpp_map(X, M, MaxIter=500, covH=cov_H, covD=cov_D_2d, linkH=link_exp_to_gauss, argsH=[np.diag(cov_H), 1],
               linkD=link_rectgauss, argsD=[np.diag(cov_D_2d), 1], sigma_N=sigma_N)


print("\nComputing LS NMF")
D_ls, H_ls = nmf_ls(X, M, maxiter=100)

a1 = mat1["a"]
a2 = mat2["a"]
vp1 = mat1["vp"]
vp2 = mat2["vp"]

true_spec1 = (a1.T@vp1)[0]
true_spec2 = (a2.T@vp2)[0]

nmf_gpp_spec1_1d = H_1d[2]
nmf_gpp_spec2_1d = H_1d[0]
nmf_gpp_noise_1d = H_1d[1]

nmf_ls_spec1 = H_ls[1]
nmf_ls_spec2 = H_ls[0]
nmf_ls_noise = H_ls[2]

nmf_gpp_spec1_2d = H_2d[0]
nmf_gpp_spec2_2d = H_2d[1]
nmf_gpp_noise_2d = H_2d[2]


file_extension = ".svg"
prefix = "./results/"

plt.figure()
plt.subplot(311)
plt.plot(true_spec1,label = "True spectrum")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(312)
plt.plot(nmf_gpp_spec1_1d,label = "NMF GPP 1d")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(313)
plt.plot(nmf_gpp_spec1_2d,label = "NMF GPP 2d")
plt.legend()
plt.xlabel("Wave number")
plt.gca().axes.get_yaxis().set_visible(False)
plt.savefig(prefix + file_name_result +'2d1dCov_spec1' + file_extension)

plt.figure()
plt.subplot(311)
plt.plot(true_spec2,label = "True spectrum")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(312)
plt.plot(nmf_gpp_spec2_1d,label = "NMF GPP 1d")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.xlabel("Wave number")
plt.legend()
plt.subplot(313)
plt.plot(nmf_gpp_spec2_2d,label = "NMF GPP 2d")
plt.xlabel("Wave number")
plt.legend()
plt.gca().axes.get_yaxis().set_visible(False)
plt.savefig(prefix + file_name_result +'2d1dCov_spec2' + file_extension)


plt.figure()
plt.subplot(311)
plt.plot(true_spec1,label = "True spectrum")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(312)
plt.plot(nmf_ls_spec1,label = "NMF LS")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(313)
plt.plot(nmf_gpp_spec1_1d,label = "NMF GPP 1d")
plt.legend()
plt.xlabel("Wave number")
plt.gca().axes.get_yaxis().set_visible(False)
plt.savefig(prefix + file_name_result +'_spec1' + file_extension)

plt.figure()
plt.subplot(311)
plt.plot(true_spec2,label = "True spectrum")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(312)
plt.plot(nmf_ls_spec2,label = "NMF LS")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(313)
plt.plot(nmf_gpp_spec2_1d,label = "NMF GPP 1d")
plt.xlabel("Wave number")
plt.legend()
plt.gca().axes.get_yaxis().set_visible(False)
plt.savefig(prefix + file_name_result +'_spec2' + file_extension)

plt.figure()
plt.subplot(211)
plt.plot(nmf_ls_noise,label = "NMF LS")
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.legend()
plt.subplot(212)
plt.plot(nmf_gpp_noise_1d,label = "NMF GPP 1d")
plt.xlabel("Wave number")
plt.legend()
plt.gca().axes.get_yaxis().set_visible(False)
plt.savefig(prefix + file_name_result +'_noise' + file_extension)