import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def LD(rel, return_inbreeding_coefficient=False):
    ancestors_list = np.split(rel.indices, rel.indptr[1:-1])
    n = rel.shape[0]
    L = lil_matrix((n, n))
    D = np.zeros((n))
    F = np.zeros((n))

    for i in range(n):

        L[i, i] = 1
        ANC = [i]
        i_parents = ancestors_list[i]
        D[i] = 1 - 0.25 * (i_parents.shape[0] + F[i_parents].sum())

        while len(ANC) > 0:
            j = max(ANC)
            j_parents = ancestors_list[j]
            ANC += j_parents.tolist()
            L[i, j_parents] += np.ones(j_parents.size) * 0.5 * L[i, j]
            F[i] += (L[i, j] ** 2) * D[j]
            ANC = [x for x in ANC if x != j]
        F[i] -= 1

    D_full = csr_matrix((n, n))
    D_full[np.arange(n), np.arange(n)] = D
    L = L.tocsr()

    if return_inbreeding_coefficient:
        return L, D_full, F
    return L, D_full


def create_numerator(L, D):
    return L.dot(D).dot(L.transpose(copy=True)).tocsr()


def simple_numerator(rel):
    L, D = LD(rel)
    return create_numerator(L, D), L, D


def selective_numerator(L, D, indices):
    # TODO: write the efficient version in it's better format
    raise NotImplemented


if __name__ == "__main__":
    # Validate
    from Examples.GraphExamples import henderson_example

    rel, real, L, D = henderson_example()
    newL, newD_squared = LD(rel)
    assert ((L - newL).nnz == 0) and (np.sqrt(newD_squared) - D).max() < 0.00001
    assert (simple_numerator(newL, newD_squared) - real).max() < 0.00001

    # Test time
    print("Simulate pedigree")
    from Simulation.Pedigree import simulate_tree

    rel, _, _ = simulate_tree(100000, 0.001, 1.4, 0.9)
    print("Computing LD")
    bigL, bigD_squared = LD(rel)
    print("Computing IBD")
    create_numerator(bigL, bigD_squared)
