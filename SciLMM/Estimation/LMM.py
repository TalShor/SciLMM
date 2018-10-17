import numpy as np
import scipy.stats as stats
import scipy.linalg as la
import scipy.optimize as optimize
import time
from sksparse.cholmod import cholesky as sk_choleksy
from itertools import combinations_with_replacement
from scipy.sparse import eye

np.set_printoptions(precision=4, linewidth=200)


class SparseCholesky(object):
    def __init__(self, use_long=False, mode='supernodal', ordering_method='nesdis'):
        self._use_long = use_long
        self._mode = mode
        self._ordering_method = ordering_method

    def __call__(self, sparse_mat):
        return sk_choleksy(sparse_mat,
                           use_long=self._use_long,
                           mode=self._mode,
                           ordering_method=self._ordering_method)



def compute_fixed_effects(factor, y, covariates):
    invV_C = factor(covariates)
    L_CT_invV_C = la.cho_factor(covariates.T.dot(invV_C))
    fixed_effects = la.cho_solve(L_CT_invV_C, covariates.T.dot(factor(y)))
    mu = covariates.dot(fixed_effects)
    return invV_C, L_CT_invV_C, mu, fixed_effects


def negative_log_likelihood(factor, y, invV_y, mu, L_CT_invV_C, reml):
    n = y.size
    nll_numer = (y - mu).dot(invV_y)
    nll_denom = (n * np.log(2 * np.pi) + factor.logdet())
    nll = 0.5 * (nll_numer + nll_denom)

    if reml:
        nll += 0.5 * 2 * np.sum(np.log(np.diag(L_CT_invV_C[0])))  # ignoring some irrelevant constants

    return nll


def simulate_vector(factor, n, sim_num, p_inv):
    sim_vec = factor.L().dot(np.random.randn(n, sim_num))
    sim_vec = sim_vec[p_inv]
    return factor(sim_vec)


def matrices_weighted_sum(mats, sig2g_array):
    V = sig2g_array[0] * mats[0]
    for i in range(1, len(sig2g_array)):
        V += sig2g_array[i] * mats[i]
    return V.tocsc()


def compute_gradients(sig2g_array, mats, sim_vec, invV_y, reml, invV_C, L_CT_invV_C):
    grad = np.zeros(len(sig2g_array))
    for grad_i in range(len(sig2g_array)):
        comp1 = np.mean(np.sum(mats[grad_i].dot(sim_vec) * sim_vec, axis=0))
        comp2 = invV_y.dot(mats[grad_i].dot(invV_y))
        grad[grad_i] = 0.5 * (comp1 - comp2)  # this is the gradient of the *negative* log likelihood

        if reml:
            vec = invV_C.T.dot(mats[grad_i].dot(invV_C))
            temp = la.cho_solve(L_CT_invV_C, vec)
            grad[grad_i] -= 0.5 * np.trace(temp)

    return grad


def bolt_gradient_estimation(log_sig2g_array, cholesky, mats, covariates, y, reml, sim_num, verbose):
    sig2g_array = np.exp(log_sig2g_array)

    if verbose:
        t0 = time.time()
        print('estimating nll and its gradient at:', sig2g_array)

    V = matrices_weighted_sum(mats, sig2g_array)
    n = V.shape[0]

    # Compute the Cholesky factorization of V, and permute all the other matrices accordingly
    factor = cholesky(V)
    P = factor.P()
    p_inv = np.argsort(P)

    # compute fixed effects
    invV_C, L_CT_invV_C, mu, _ = compute_fixed_effects(factor, y, covariates)

    # compute invV.dot(y-mu)
    invV_y = factor(y - mu)

    # compute the negative log likelihood
    nll = negative_log_likelihood(factor, y, invV_y, mu, L_CT_invV_C, reml)

    # simulate vectors for the BOLT-LMM trick
    sim_vec = simulate_vector(factor, n, sim_num, p_inv)

    # compute the gradient with respect to log_sig2g (instead of sig2g directly), using the chain rule
    grad = compute_gradients(sig2g_array, mats, sim_vec, invV_y, reml, invV_C, L_CT_invV_C)
    grad *= sig2g_array

    if verbose:
        print("grad : ", grad)
        print('nll: %0.8e   computation time: %0.2f seconds' % (nll, time.time() - t0))

    return nll, grad


def compute_sigmas(cholesky, mats, covariates, y, reml=True, sim_num=100, verbose=True):
    # randomly choose a valid starting point
    x0 = np.ones((len(mats)))# = np.random.random(len(mats));
    x0 = np.log(x0 / x0.sum())

    # minimize the likelihood (REML or not)
    optObj_bfgs = optimize.minimize(bolt_gradient_estimation, x0,
                                    args=(cholesky, mats, covariates, y, reml, sim_num, verbose),
                                    jac=True, method='L-BFGS-B',
                                    options={'eps': 1e-5, 'ftol': 1e-7})

    mats_coefficients = np.exp(optObj_bfgs.x)

    return mats_coefficients


def compute_fixed_effects_p_value(y, covariates, fixed_effects, L_CT_invV_C):
    var_fixedeffects = la.cho_solve(L_CT_invV_C, np.eye(covariates.shape[1]))
    test_stats = fixed_effects ** 2 / np.diag(var_fixedeffects)
    p_values = stats.f(1, y.shape[0] - 1).sf(test_stats)
    return p_values


def compute_sig_of_sig(mats, covariates, factor, y, sim_num):
    num_matrices = len(mats)

    V_inv_y = factor(y)
    V_inv_C = factor(covariates)
    C_t_V_inv_C__inv = np.linalg.inv(covariates.T.dot(V_inv_C))
    P = V_inv_y - V_inv_C.dot(C_t_V_inv_C__inv.dot(covariates.T.dot(V_inv_y)))

    hess = np.empty((num_matrices, num_matrices))
    factor_array = [factor(mats[j].dot(P)) for j in range(num_matrices)]
    for i, j in combinations_with_replacement(range(num_matrices), 2):
        hess[i, j] = -0.5 * y.T.dot(factor(mats[i].dot(factor_array[j])))
        hess[j, i] = hess[i, j]

    inv_neg_hess = la.inv(-hess)
    return np.sqrt((inv_neg_hess * (1 + 1.0 / sim_num))[range(num_matrices), range(num_matrices)])


def LMM(cholesky, mats, covariates, y, with_intercept=True, reml=True, sim_num=100, verbose=False):
    mats = mats + [eye(y.size).tocsr()]
    if with_intercept:
        covariates = np.hstack((np.ones((y.size, 1)), covariates))
    mats_coefficients = compute_sigmas(cholesky, mats, covariates, y, reml, sim_num, verbose)
    V = matrices_weighted_sum(mats, mats_coefficients)
    factor = cholesky(V)

    _, L_CT_invV_C, _, fixed_effects = compute_fixed_effects(factor, y, covariates)
    fixed_effects_p_values = compute_fixed_effects_p_value(y, covariates, fixed_effects, L_CT_invV_C)

    sigmas_sigmas = compute_sig_of_sig(mats, covariates, factor, y, sim_num)
    return {"covariance coefficients": mats_coefficients,
            "covariates coefficients": fixed_effects,
            "covariance std": sigmas_sigmas,
            "covariates p-values": fixed_effects_p_values}


if __name__ == "__main__":
    from Simulation.Pedigree import simulate_tree
    from Simulation.Phenotype import simulate_phenotype
    from Matrices.Numerator import simple_numerator
    from Matrices.Epistasis import pairwise_epistasis
    from Matrices.Dominance import dominance

    rel, _, _ = simulate_tree(50000, 0.001, 1.4, 0.9)
    ibd, _, _ = simple_numerator(rel)
    epis = pairwise_epistasis(ibd)
    dom = dominance(rel, ibd)
    cov = np.random.randn(50000, 2)
    y = simulate_phenotype([ibd, epis, dom],
                           cov,
                           np.array([0.3, 0.2, 0.1, 0.4]),
                           np.array([0.01, 0.02, 0.03]),
                           True)
    print (LMM(SparseCholesky(), [ibd, epis, dom], cov, y))
