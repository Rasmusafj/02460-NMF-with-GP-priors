from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from core import nmf_ls, nmf_gpp_map, link_exp_to_gauss, link_rectgauss, rbf, get_2d_rbf_kernel
file_name1 = '50x50_nw250_nhs2_k4_1'
mat1 = loadmat('../data/'+file_name1+'.mat')
X1 = mat1["X"]

file_name2 = '50x50_nw250_nhs2_k4_2'
mat2 = loadmat('../data/'+file_name2+'.mat')
X2 = mat2["X"]

X = X1 + X2

a = mat1["a"] + mat2["a"]
vp = mat1["vp"] + mat2["vp"]

file_name_result = '2_spec_50x50_nw250_nhs2_k4'


K, L = X.shape
M = 3
beta_H = 10
beta_D = 10
sigma_N = 5
np.random.seed(125)

#cov_D = np.zeros((K, K))
#for i in range(K):
#   for j in range(K):
#       cov_D[i, j] = rbf(beta_D, i + 1, j + 1)

dim = int(sqrt(len(X)))
cov_D = get_2d_rbf_kernel(beta_D, (dim,dim))
cov_D = cov_D + 1e-5 * np.eye(K)


cov_H = np.zeros((L, L))
for i in range(L):
    for j in range(L):
        cov_H[i, j] = rbf(beta_H, i + 1, j + 1)

cov_H = cov_H + 1e-5 * np.eye(L)

cholD = np.linalg.cholesky(cov_D)
cholH = np.linalg.cholesky(cov_H)

print("\nComputing GPP NMF with correct prior.")
D, H = nmf_gpp_map(X, M, MaxIter=500, covH=cov_H, covD=cov_D, linkH=link_exp_to_gauss, argsH=[np.diag(cov_H), 1],
               linkD=link_rectgauss, argsD=[np.diag(cov_D), 1], sigma_N=sigma_N - 3)


X_re = D.T @ H

D_ls, H_ls = nmf_ls(X, M, maxiter=10000)

X_re_ls = D_ls @ H_ls


a1 = mat1["a"]
a2 = mat2["a"]
vp1 = mat1["vp"]
vp2 = mat2["vp"]

true_spec1 = (a1.T@vp1)[0]
true_spec2 = (a2.T@vp2)[0]
true_spec1 = true_spec1/max(true_spec1)
true_spec2 =true_spec2/max(true_spec2)


nmf_spec1 = H_ls[0]
nmf_spec2 = H_ls[1]
nmf_spec1 /=max(nmf_spec1)
nmf_spec2 /= max(nmf_spec2)

nmf_gpp_spec1 = H[2]
nmf_gpp_spec2 = H[0]
nmf_gpp_spec1 /= max(nmf_gpp_spec1)
nmf_gpp_spec2 /= max(nmf_gpp_spec2)

plt.figure()
plt.plot(true_spec1, label = "True spectrum")
plt.plot(nmf_spec1, label = "NMF LS")
plt.plot(nmf_gpp_spec1, label = "NMF GPP")
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('/Results/'+file_name_result+'2dCov_spec1.png')
plt.show()

plt.figure()
plt.plot(true_spec2, label = "True spectrum")
plt.plot(nmf_spec2, label = "NMF LS")
plt.plot(nmf_gpp_spec2, label = "NMF GPP")
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('/Results/'+file_name_result+'2dCov_spec2.png')

plt.figure()
plt.plot(true_spec1+true_spec2, label = "True spectrum")
plt.plot(nmf_spec1+nmf_spec2, label = "NMF LS")
plt.plot(nmf_gpp_spec1+nmf_gpp_spec2, label = "NMF GPP")
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('/Results/'+file_name_result+'2dCov_full_spec.png')



