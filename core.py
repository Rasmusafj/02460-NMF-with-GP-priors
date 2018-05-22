import numpy as np
from scipy.special import erf, erfinv
from scipy.optimize import minimize
from tqdm import tqdm
import pymc3 as pm
import sys
import theano.tensor as tt

from numpy.matlib import repmat, repeat
from custom_pymc_implementation import CustomLatent

########################
#### LINK FUNCTIONS ####
########################

def link_exp_to_gauss(h, pars):
    sigma = pars[0]
    lamb = pars[1]  # inverse scale
    # actual inverse link value
    inner = .5 - .5*erf(h / (np.sqrt(2) * sigma)) + 1e-12
    val = np.maximum(-1 / lamb * np.log(inner), 0)
    val = np.nan_to_num(val)
    # elementwise derivative of inverse link
    grad = (np.sqrt(2 * np.pi) * sigma * lamb) ** (-1) * np.exp(lamb * val - h ** 2 / (2 * sigma ** 2))
    return val, grad


def link_rectgauss(h, pars):
    sigma = pars[0]  # diag of Sigma_h
    s = pars[1]  # "width" parameter
    # value of inverse link
    inner = .5 + .5 * erf(h / (np.sqrt(2) * sigma))
    val = np.sqrt(2) * s * erfinv(inner)

    # elementwise derivative of inverse link
    grad = s / (2 * sigma) * np.exp(val ** 2 / (2 * s ** 2) - h ** 2 / (2 * sigma ** 2))

    return val, grad


########################
######## kernels #######
########################

def get_2d_rbf_kernel(beta, shape_plate):
    dummy = np.ones(shape_plate)
    dummy = np.argwhere(dummy)
    distances = calcDistanceMatrixFastEuclidean(dummy) ** 2
    final = np.exp((-distances) / (beta ** 2))

    return final


def calcDistanceMatrixFastEuclidean(points):
    numPoints = len(points)
    distMat = np.sqrt(np.sum((repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2, axis=1))
    return distMat.reshape((numPoints,numPoints))


def rbf(beta, i, j):
    return np.exp(- ((i - j) ** 2 / (beta ** 2)))


############################
###### Pymc3 sampling ######
############################

def loglik_X(X, d, h):
    """
    Hard coded log likelihood in the form of tensors

    :param X: Observed
    :param d: Prior
    :param h: Prior
    :return: The loss
    """
    sigma_N = 5
    lamb = 1
    sigmas = 1
    s = 1
    M = 2

    inner = .5 - .5 * pm.math.erf(h / (pm.math.sqrt(2) * sigmas)) + 1e-12
    H = pm.math.maximum(-1 / lamb * pm.math.log(inner), 0)

    inner = .5 + .5 * pm.math.erf(d / (pm.math.sqrt(2) * sigmas))
    D = pm.math.sqrt(2) * s * pm.math.erfinv(inner)
    sh = X.value.shape
    D = tt.reshape(D, (sh[0], M))
    H = tt.reshape(H, (M, sh[1]))

    return - 1 / (sigma_N * sigma_N) * pm.math.sum(pm.math.sum((X - pm.math.dot(D, H)) * (X - pm.math.dot(D, H))))


def nmf_gpp_hmc(X, M, **kwargs):
    """
    Samples posterior of NMF GPP with Hamiltonian Monte Carlo using the Leapfrog method for .
    :param X: Data matrix
    :param M: Number of latent factors (ie. sources)
    :param kwargs:  Options:
                    'numSamples': Number of samples to be drawn
                    'linkH'     : Link function for H, callable. Inverse link
                    'argsH'     : Extra arguments for linkH. Should be a list.
                    'linkD'     : Link function for D, callable. Inverse link
                    'argsD'     : Extra arguments for linkD. Should be a list.
                    'sigma_N'   : Variance of Gaussian noise.
                    'burn'      : Burn-in to be discarded
                    'dimD'      : Dimension of covariance for D
                    'dimH'      : Dimension of covariance for H

    :return: Traces of NN matrix factors, D and H
    """
    # parse arguments
    try:
        numSamples = kwargs['numSamples']
        dimD = kwargs['dimD']
        dimH = kwargs['dimH']
        numChains = kwargs['numChains']
        db_name = kwargs['db_name']
    except KeyError:
        print("Missing parameter with no default. Terminating")
        sys.exit(1)

    K, L = X.shape

    d_in = np.arange(K*M)[:, None]
    h_in = np.arange(M*L)[:, None]

    # custom likelihood for X given d and h.
    # TODO: Find log-likelihood parameterized by X, d and h. Eq. (4) with reshape and link function

    # begin actual model
    with pm.Model() as mod:
        ls_D = pm.Gamma(name='lsd', alpha=3, beta=1, shape=(dimD,))
        #covD = pm.gp.cov.ExpQuad(input_dim=dimD, ls=ls_D, active_dims=400)
        covD = pm.gp.cov.Exponential(input_dim=dimD, ls=ls_D)
        gpD = CustomLatent(cov_func=covD)
        d = gpD.prior("d", X=d_in.reshape(K, M))

        ls_H = pm.Gamma(name='lsh', alpha=3, beta=1, shape=(dimH,))
        covH = pm.gp.cov.ExpQuad(input_dim=dimH, ls=ls_H)
        gpH = CustomLatent(cov_func=covH)
        h = gpH.prior("h", X=h_in.reshape(L, M))

        X_ = pm.DensityDist('X', loglik_X, observed={'X': X,
                                                     'd': d,
                                                     'h': h})

        db = pm.backends.Text(db_name)
        trace = pm.sample(numSamples, njobs=1, trace=db, chains=numChains, tune=5)

    return trace


###################################
#### NMF-GPP with map estimate ####
###################################

def nmf_gpp_map(X, M, **kwargs):
    """
    Computes the MAP estimate of Non-negative matrix factorization with Gaussian Process Priors.
    :param X:       Data matrix, K x L
    :param M:       Number of factors.
    :param kwargs:  Options:
                    'MaxIter'   : Iterations for optimization
                    'covH'      : Covariance of GPP determining H, M*L x M*L
                    'covD'      : Covariance of GPP determining D, K*M x K*M
                    'linkH'     : Link function for H, callable. Inverse link
                    'argsH'     : Extra arguments for linkH. Should be a list.
                    'linkD'     : Link function for D, callable. Inverse link
                    'argsD'     : Extra arguments for linkD. Should be a list.
                    'sigma_N'   : Variance of Gaussian noise.

    :return D,H:    NN Matrix Factors
    """
    # parse arguments
    try:
        maxIter = kwargs['MaxIter']
        covH = kwargs['covH']
        covD = kwargs['covD']
        linkH = kwargs['linkH']
        argsH = kwargs['argsH']
        linkD = kwargs['linkD']
        argsD = kwargs['argsD']
        sigma_N = kwargs['sigma_N']
    except KeyError:
        print("Missing parameter(s). All parameters must be passed.")
        return None
    K, L = X.shape
    # initial value of delta, eta
    delta = np.random.randn(M*K)
    eta = np.random.randn(M*L)

    # compute D and H from delta and eta
    def cost(opti, other, sigma, X, ret_eta, linkD, linkH, M, cholD, cholH, *linkArgs):
        # linkD and linkH should be callable and return f^-1(h) and f^-1(d) resp.
        # furthermore it should return the derivative of these!
        # linkArgs is a nested list where the first list is a list of extra args to linkD
        # and second list is a list of extra args to linkD
        K, L = X.shape
        if ret_eta:
            eta = opti
            delta = other
        else:
            delta = opti
            eta = other
        # convert 'greeks' to matrices D and H - eq. (17)
        D_args = linkArgs[0]
        # link_rectgauss
        D, D_prime = linkD((delta.reshape(M, K) @ cholD.T), D_args)

        H_args = linkArgs[1]
        # link_exp_to_gauss
        H, H_prime = linkH((eta.reshape(M, L) @ cholH.T), H_args)

        # cost itself - eq. (22)
        first = sigma ** (-2) * np.sum(np.sum((X - D.T @ H) * (X - D.T @ H)))
        second = (delta.T @ delta) + (eta.T @ eta)
        cost_val = .5 * (first + second)

        X_re = D.T @ H
        # gradient of cost - eq. (23)
        if ret_eta:
            inner = ((D @ (X_re - X)) * H_prime.reshape(M, L))
            grad = sigma ** (-2) * (inner @ cholH).ravel() + eta
        else:
            inner = ((H @ (X_re - X).T) * D_prime.reshape(M, K))
            grad = sigma ** (-2) * (inner @ cholD).ravel() + delta
        return cost_val, grad

    # iteratively optimize over eta and delta
    cholD = np.linalg.cholesky(covD)
    cholH = np.linalg.cholesky(covH)

    for _ in tqdm(range(maxIter)):

        # optimize delta
        delta_old = delta
        args = (eta, sigma_N, X, False, linkD, linkH, M, cholD, cholH, argsD, argsH)
        delta = minimize(cost, delta_old, args, 'L-BFGS-B', jac=True).x


        # optimize eta
        eta_old = eta
        args = (delta, sigma_N, X, True, linkD, linkH, M, cholD, cholH, argsD, argsH)
        eta = minimize(cost, eta_old, args, 'L-BFGS-B', jac=True).x


    #  convert to D and H
    D = linkD(delta.reshape(M, K) @ cholD.T, argsD)[0]
    H = linkH(eta.reshape(M, L) @ cholH.T, argsH)[0]

    return D, H


############################
######## NMF-LS ############
############################

def nmf_ls(X, M, maxiter=1000):
    K, L = X.shape
    # D is K x M
    # H is M x L
    D = np.random.uniform(1e-12, 100, (K, M))
    H = np.random.uniform(1e-12, 100, (M, L))

    # as the updates are iterative, we have to ensure that X is already non-negative.
    X_nn = X.copy()
    X_nn[X_nn < 0] = 0.0

    for i in range(maxiter):
        Dt = D.T * (H @ X_nn.T) / (H @ H.T @ D.T)
        Dt = Dt / np.sum(Dt, axis=0)
        D = Dt.T
        H = H * (Dt @ X_nn) / (Dt @ D @ H)

    return D, H