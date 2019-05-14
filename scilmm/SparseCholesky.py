import time

import numpy as np
import pandas as pd
import scipy.linalg as la
import scipy.optimize as optimize
import scipy.sparse
from scipy.io import mmread
from sksparse.cholmod import cholesky as sk_cholesky

np.set_printoptions(precision=3, linewidth=200)
pd.set_option('display.width', 200)


class SparseCholesky(object):
    def __init__(self, use_long=False, mode='supernodal', ordering_method='nesdis'):
        self._use_long = use_long
        self._mode = mode
        self._ordering_method = ordering_method

    def __call__(self, sparse_mat):
        return sk_cholesky(sparse_mat,
                           use_long=self._use_long,
                           mode=self._mode,
                           ordering_method=self._ordering_method)


def estimate_fixed_effects(factor, y, covariates):
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


def bolt_gradient_estimation(log_sig2g_array, cholesky_func, mats, covariates, y, reml, sim_num, verbose,
                             take_exp=True):
    if take_exp:
        sig2g_array = np.exp(log_sig2g_array)
    else:
        sig2g_array = log_sig2g_array

    if verbose:
        t0 = time.time()
        print('estimating nll and its gradient at:', sig2g_array)

    V = matrices_weighted_sum(mats, sig2g_array)
    n = V.shape[0]

    # Compute the Cholesky factorization of V, and permute all the other matrices accordingly
    factor = cholesky_func(V)
    P = factor.P()
    P_inv = np.argsort(P)

    # compute fixed effects
    invV_C, L_CT_invV_C, mu, _ = estimate_fixed_effects(factor, y, covariates)

    # compute invV.dot(y-mu)
    invV_y = factor(y - mu)

    # compute the negative log likelihood
    nll = negative_log_likelihood(factor, y, invV_y, mu, L_CT_invV_C, reml)

    # simulate vectors for the BOLT-LMM trick
    sim_vec = simulate_vector(factor, n, sim_num, P_inv)

    # compute the gradient with respect to log_sig2g (instead of sig2g directly), using the chain rule
    grad = compute_gradients(sig2g_array, mats, sim_vec, invV_y, reml, invV_C, L_CT_invV_C)
    if take_exp:
        grad *= sig2g_array

    if verbose:
        print("grad : ", grad)
        print('nll: %0.8e   computation time: %0.2f seconds' % (nll, time.time() - t0))

    return nll, grad


def estimate_var_comps(cholesky_func, mats, covariates, y, reml=True, sim_num=100, verbose=True, aireml=False):
    he_est = HE(cholesky_func, mats[:-1], covariates, y, compute_stderr=False)
    he_est = np.concatenate((he_est, [1 - he_est.sum()]))
    x0 = he_est
    if np.any(x0 < 0):
        x0 = np.ones((len(mats)))
    x0 /= x0.sum()

    # AI-REML algorithm
    if aireml:
        raise NotImplementedError('AI-REML is broken')

    # L-BFGS-B optimization
    else:
        log_x0 = np.log(x0)
        optObj = optimize.minimize(bolt_gradient_estimation, log_x0,
                                   args=(cholesky_func, mats, covariates, y, reml, sim_num, verbose, True),
                                   jac=True, method='L-BFGS-B',
                                   options={'eps': 1e-5, 'ftol': 1e-7})
    if not optObj.success:
        print('optimization failed with message: %s' % optObj.message)

    mats_coefficients = np.exp(optObj.x)

    return mats_coefficients


def compute_hess(mats, covariates, factor, y):
    num_matrices = len(mats)
    Vinv_C = factor(covariates)
    L_CT_Vinv_C = la.cho_factor(covariates.T.dot(Vinv_C))

    def P(z):
        Vinv_z = factor(z)
        Pz = Vinv_z - Vinv_C.dot(la.cho_solve(L_CT_Vinv_C, covariates.T.dot(Vinv_z)))
        return Pz

    Py = P(y)
    hess = np.empty((num_matrices, num_matrices))
    for j in range(num_matrices):
        Hj_Py = mats[j].dot(Py)
        P_Hj_Py = P(Hj_Py)
        for i in range(j + 1):
            Hi_P_Hj_Py = mats[i].dot(P_Hj_Py)
            P_Hi_P_Hj_Py = P(Hi_P_Hj_Py)
            hess[i, j] = -0.5 * y.dot(P_Hi_P_Hj_Py)
            hess[j, i] = hess[i, j]

    return hess


def compute_varcomp_stderr(mats, covariates, factor, y, sim_num):
    hess = compute_hess(mats, covariates, factor, y)
    inv_neg_hess = la.inv(-hess)
    return np.sqrt(np.diag(inv_neg_hess) * (1 + 1.0 / sim_num))


def REML(cholesky_func, mats, covariates, y, reml=True, sim_num=100, verbose=False):
    y = y / y.std()
    mats = mats + [scipy.sparse.eye(y.shape[0]).tocsr()]
    varcomp_estimates = estimate_var_comps(cholesky_func, mats, covariates, y, reml, sim_num, verbose)
    V = matrices_weighted_sum(mats, varcomp_estimates)
    factor = cholesky_func(V)

    _, _, _, fixed_effects = estimate_fixed_effects(factor, y, covariates)

    sigmas_sigmas = compute_varcomp_stderr(mats, covariates, factor, y, sim_num)
    return {"covariance coefficients": varcomp_estimates,
            "covariates coefficients": fixed_effects,
            "covariance std": sigmas_sigmas}


def HE(cholesky_func, mat_list, cov, y, MQS=True, verbose=False, sim_num=100, compute_stderr=False):
    # regress all covariates out of y
    CTC = cov.T.dot(cov)
    y = y - cov.dot(np.linalg.solve(CTC, cov.T.dot(y)))

    # standardize y
    y /= y.std()
    # assert np.isclose(y.mean(), 0)
    # assert np.isclose(y.var(), 1)
    K = len(mat_list)

    # construct S and q, without MQS
    if not MQS:
        q = np.zeros(K)
        S = np.zeros((K, K))
        for i, mat_i in enumerate(mat_list):
            if scipy.sparse.issparse(mat_i):
                # q[i] = ((mat_i.multiply(y)).T.tocsr().dot(y)).sum() - mat_i.diagonal().dot(y**2)
                q[i] = y.dot(mat_i.dot(y)) - mat_i.diagonal().dot(y ** 2)
            else:
                q[i] = ((mat_i * y).T.dot(y)).sum() - np.diag(mat_i).dot(y ** 2)
            for j, mat_j in enumerate(mat_list):
                if j > i: continue
                if scipy.sparse.issparse(mat_i):
                    S[i, j] = (mat_i.multiply(mat_j)).sum() - mat_i.diagonal().dot(mat_j.diagonal())
                else:
                    S[i, j] = np.einsum('ij,ij->', mat_i, mat_j) - np.diag(mat_i).dot(np.diag(mat_j))
                S[j, i] = S[i, j]

                # construct S and q with MQS (it's almost the same thing...)
    else:
        n = y.shape[0]
        q = np.zeros(K)
        S = np.zeros((K, K))
        for i, mat_i in enumerate(mat_list):
            q[i] = (y.dot(mat_i.dot(y)) - y.dot(y))  # / float(n-1)**2
            for j, mat_j in enumerate(mat_list):
                if j > i: continue
                S[i, j] = ((mat_i.multiply(mat_j)).sum() - (n - 1))  # / float(n-1)**2
                S[j, i] = S[i, j]

    # compute HE
    he_est = np.linalg.solve(S, q)

    if not compute_stderr:
        return he_est

    # compute H - the covariance matrix of y
    H = mat_list[0] * he_est[0]
    for mat_i, sigma2_i in zip(mat_list[1:], he_est[1:]):
        H += mat_i * he_est[i]
    H += scipy.sparse.eye(y.shape[0], format='csr') * (1.0 - he_est.sum())

    # compute HE sampling variance
    V_q = np.empty((K, K))
    for i, mat_i in enumerate(mat_list):
        if sim_num is None:
            HAi_min_I = H.dot(mat_i) - H
        for j, mat_i in enumerate(mat_list[:i + 1]):
            if sim_num is None:
                if j == i:
                    HAj_min_I = HAi_min_I
                else:
                    HAj_min_I = H.dot(mat_j) - H
                V_q[i, j] = 2 * (HAi_min_I.multiply(HAj_min_I)).sum()  # / float(n-1)**4
            else:
                # simulate vectors
                assert cholesky_func is not None
                sim_y = np.random.randn(n, sim_num)
                Aj_minI_y = mat_j.dot(sim_y) - sim_y
                H_Aj_minI_y = H.dot(Aj_minI_y)
                Ai_min_I_H_Aj_minI_y = mat_i.dot(H_Aj_minI_y) - H_Aj_minI_y
                H_Ai_min_I_H_Aj_minI_y = H.dot(Ai_min_I_H_Aj_minI_y)
                V_q[i, j] = 2 * np.mean(np.einsum('ij,ij->j', sim_y, H_Ai_min_I_H_Aj_minI_y))

            V_q[j, i] = V_q[i, j]
    var_he_est = np.linalg.solve(S, np.linalg.solve(S, V_q).T).T

    return he_est, np.sqrt(np.diag(var_he_est))


def MINQUE(cholesky_func, mat_list, cov, y, compute_stderr=False, verbose=False, num_iter=100, sim_num=100):
    # regress all covariates out of y
    CTC = cov.T.dot(cov)
    y = y - cov.dot(np.linalg.solve(CTC, cov.T.dot(y)))

    # standardize y
    y /= y.std()
    assert np.isclose(y.mean(), 0)
    assert np.isclose(y.var(), 1)
    K = len(mat_list)

    H = None  # None means identity
    n = y.shape[0]
    for iter_num in range(num_iter):
        q = np.zeros(K)
        S = np.zeros((K, K))

        # decompose H and sample normal vectors with covariance matrix H
        if H is not None:
            factor = cholesky_func(H)
            sim_y = factor.L().dot(np.random.randn(n, sim_num))
            sim_y = sim_y[np.argsort(factor.P())]
            invH_simy = factor(sim_y)

        for i, mat_i in enumerate(mat_list):
            # construct q_i
            if H is None:
                q[i] = y.dot(mat_i.dot(y)) - y.dot(y)
            else:
                invH_y = factor(y)
                q[i] = invH_y.dot(mat_i.dot(invH_y)) - y.dot(y)
                Ki_invH_simy = mat_i.dot(invH_simy)
                invH_Ki_invH_simy = factor(Ki_invH_simy)

            # construct S_ij
            for j, mat_j in enumerate(mat_list):
                if (j > i): continue
                if H is None:
                    S[i, j] = (mat_i.multiply(mat_j)).sum() - (n - 1)
                else:
                    Kj_invH_Ki_invH_simy = mat_j.dot(invH_Ki_invH_simy)
                    S[i, j] = np.mean(np.einsum('ij,ij->j', invH_simy, Kj_invH_Ki_invH_simy)) - (n - 1)
                S[j, i] = S[i, j]

        # compute minque_est
        minque_est = np.linalg.solve(S, q)
        print(iter_num + 1, minque_est)

        # compute H
        H = mat_list[0] * minque_est[0]
        for mat_i, sigma2_i in zip(mat_list[1:], minque_est[1:]):
            H += mat_i * minque_est[i]
        H += scipy.sparse.eye(y.shape[0], format='csr') * (1.0 - minque_est.sum())

    # compute MINQUE
    minque_est = np.linalg.solve(S, q)

    if not compute_stderr:
        return minque_est

    # var_he_est = np.linalg.solve(S, np.linalg.solve(S, V_q).T).T
    var_he_est = 0

    return minque_est, np.sqrt(var_he_est)


def run_estimates(A, df_phe, df_cov, reml=False, ignore_indices=False):
    # Sort shared indices between matrices:
    if not ignore_indices:
        indices = list(set(df_cov.index) & set(df_phe.index) & set(A.indices))

        # Sort all objects in the same order by IIDs
        df_cov = df_cov.loc[indices]
        df_phe = df_phe.loc[indices]
        A = A[indices][:, indices]

    # Remove individuals who have no family relationships in the data
    has_relatives = np.asarray(A.sum(axis=1))[:, 0] > 1
    if any(~has_relatives):
        A = A[has_relatives][:, has_relatives]
        df_cov = df_cov.loc[has_relatives]
        df_phe = df_phe.loc[has_relatives]
    A.eliminate_zeros()
    y = df_phe.values.reshape(-1)

    # standardize covariates (for numerical stability)
    if type(df_cov) == pd.Series:
        df_cov = df_cov.to_frame()
    df_cov['intercept'] = 1
    cov = df_cov.values.copy().astype(np.float)
    cov[:, :-1] -= cov[:, :-1].mean(axis=0)
    cov[:, :-1] /= cov[:, :-1].std(axis=0)

    if reml:
        reml_d = REML(SparseCholesky(), [A], cov, y, verbose=True)
        print(f"reml d are {reml_d[0]} and {reml_d[1]}")
        return reml_d
    else:
        he_est = HE(SparseCholesky(), [A], cov, y, compute_stderr=True)
        print(f"HE estimates are {he_est[0]} and {he_est[1]}")
        return he_est


def run_estimates_from_paths(A, phe, cov, reml=False, ignore_indices=False):
    A = mmread(A).tocsr()
    index_col = None if ignore_indices else 0
    df_cov = pd.read_csv(cov, index_col=index_col)
    df_phe = pd.read_csv(phe, header=None, index_col=index_col)
    run_estimates(A, df_phe, df_cov, reml=reml, ignore_indices=ignore_indices)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--A', required=True,
                        help="Absolute path of the covariance matrix A. This matrix should be saved as an mmatrix.")
    parser.add_argument('--phe', required=True,
                        help="Absolute path of phenotype CSV file with no header."
                             " This file should consist of two columns: IID and Phenotype."
                             " In case of 'ignore_indices'=True there should only be one column of phenotype.")
    parser.add_argument('--cov', required=True,
                        help="Absolute path of covariates CSV file with a header."
                             " This file should contain all covariates that "
                             "should be taken into account in calculation."
                             " Notice that the first column should be the IID, unless the 'ignore_indices'"
                             " flag is given at which case the column should not appear.")
    parser.add_argument('--reml', default=False, action='store_true',
                        help="Compute using REML, default case uses the HE estimation method.")
    parser.add_argument('--ignore_indices', default=False, action='store_true',
                        help="Ignore indices of individuals."
                             " This assumes that the A matrix, phenotype matrix and "
                             "covariates matrix are all ordered in the same order of individuals.")
    args = parser.parse_args()

    run_estimates_from_paths(**(args.__dict__))
