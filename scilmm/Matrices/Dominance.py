import numpy as np
from scipy.sparse import csr_matrix, eye


def add_root(mat):
    return csr_matrix((mat.data,
                       mat.indices + 1,
                       np.insert(mat.indptr, 0, 0)),
                      shape=(mat.shape[0] + 1, mat.shape[0] + 1))


def dominance(rel, ibd):
    # counting nonzero creates problems if it has zero entries.
    ibd.eliminate_zeros()

    # create a nx2 matrix for all the parents
    ancestors_list = np.split(rel.indices, rel.indptr[1:-1])
    ancestors_list_with_root = np.zeros((len(ancestors_list) + 1, 2))
    for i, anc in enumerate(ancestors_list):
        ancestors_list_with_root[i + 1][0:anc.size] = anc + 1

    # get indices etc for the new root ibd
    ibd_with_root = add_root(ibd)
    i_entries, j_entries = ibd_with_root.nonzero()
    i_parents = ancestors_list_with_root[i_entries]
    j_parents = ancestors_list_with_root[j_entries]

    # compute dominance
    dominance_values = (ibd_with_root[i_parents[:, 0], j_parents[:, 0]].A1 *
                        ibd_with_root[i_parents[:, 1], j_parents[:, 1]].A1) + \
                       (ibd_with_root[i_parents[:, 0], j_parents[:, 1]].A1 *
                        ibd_with_root[i_parents[:, 1], j_parents[:, 0]].A1)
    dominance_values *= 0.25
    dom_mat = csr_matrix((dominance_values,
                          ibd_with_root.indices,
                          ibd_with_root.indptr),
                         shape=ibd_with_root.shape)

    # remove root
    dom_mat = dom_mat[1:][:, 1:]
    dom_mat -= dom_mat.multiply(eye(ibd.shape[0]))
    dom_mat += eye(ibd.shape[0])
    return dom_mat.tocsr()


if __name__ == "__main__":
    from Examples.GraphExamples import henderson_example

    rel, ibd, L, D = henderson_example()
    print(dominance(rel, ibd))

    from Simulation.Pedigree import simulate_tree
    from Numerator import simple_numerator, LD

    print("Find pedigree")
    rel, _, _ = simulate_tree(10000, 0.001, 1.4, 0.9)
    print("Compute ibd")
    ibd = simple_numerator(*LD(rel))
    print("Compute dominance")
    dom = dominance(rel, ibd)
    x = np.linalg.eigvals(dom.todense())
    print(np.count_nonzero(x < 0), np.count_nonzero(x == 0))
    print("end")
