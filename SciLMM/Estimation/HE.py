from sklearn import linear_model
import numpy as np
from scipy.sparse import eye
from itertools import product


def regress_beta_out(y, covariates, fit_intercept):
    """

    :param y: a vector of size n (per individual)
    :param covariates: nxc matrix - n number of individuals, c number of covariates
    :param fit_intercept: Boolean
    :return: y, with the covariates regressed out
    """
    regr = linear_model.LinearRegression(fit_intercept=fit_intercept)
    regr.fit(covariates, y)
    coefs = regr.coef_.tolist() + ([regr.intercept_.tolist()] if fit_intercept else [])
    return y - regr.predict(covariates), coefs


def compute_HE(y, covariates, covariance_matrices, fit_intercept=False):
    n = covariance_matrices[0].shape[0]
    m = len(covariance_matrices)
    y, cov_coefs = regress_beta_out(y, covariates, fit_intercept)
    mats = [mat - mat.multiply(eye(n)) for mat in covariance_matrices]

    xtx = np.zeros((m, m))
    for i, j in product(range(m), range(m)):
        if i <= j:
            xtx[i, j] = mats[i].multiply(mats[j]).sum()
            xtx[j, i] = xtx[i, j]

    xty = np.zeros((m))
    for i in range(m):
        xty[i] = (y * mats[i]).dot(y)

    coef = np.linalg.inv(xtx).dot(xty)
    coef = np.append(coef, 1 - coef.sum())
    return coef, cov_coefs


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
    print(compute_HE(y, cov, [ibd, epis, dom], True))
