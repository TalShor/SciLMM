from sksparse.cholmod import cholesky
import numpy as np


def simulate_phenotype(covariance_matrices, covariate_matrix, sigmas, fixed_effects, add_intercept=False):
    # variance
    n = covariance_matrices[0].shape[0]

    factors = [cholesky(mat) for mat in covariance_matrices]
    sim = [factor.L().dot(np.random.randn(n)) for i, factor in enumerate(factors)]
    sim = [sim[i][np.argsort(factor.P())] for i, factor in enumerate(factors)]
    sim += [np.random.randn(n)] # simulating I
    sim = np.array(sim).T
    sim = (sim - sim.mean(axis=0)) / sim.std(axis=0)
    y = sim.dot(np.sqrt(np.array(sigmas)))

    if add_intercept:
        covariate_matrix = np.hstack((covariate_matrix, np.ones((n, 1))))

    y += covariate_matrix.dot(fixed_effects)
    return (y - y.mean()) / y.std()
