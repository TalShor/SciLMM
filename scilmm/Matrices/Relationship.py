from itertools import chain

import networkx as nx
import numpy as np
from scipy.sparse import eye


def topo_sort(rel, remove_cycles=False, check_num_parents=False):
    """
    Sort relationship csr_matrix by ancestry order.
    :param rel: relationship csr_matrix
    :param remove_cycles: boolean of whether to check for and remove cycles (unfeasible).
    :param check_num_parents: boolean of whether to check for and remove cases of more than 2 parents (unfeasible).
    :return: Sorted relationship csr_matrix, sort_order
    """
    graph = nx.DiGraph(rel)
    if remove_cycles:
        cycle_iids = nx.algorithms.cycles.recursive_simple_cycles(graph)
        if len(cycle_iids) > 0:
            edges_to_remove = list(
                map(lambda cyc: list(map(lambda i: (cyc[i], cyc[(i + 1) % len(cyc)]), range(len(cyc)))), cycle_iids))
            edges_to_remove = list(chain.from_iterable(edges_to_remove))
            graph.remove_edges_from(edges_to_remove)
            graph.remove_nodes_from(list(nx.isolates(graph)))
    if check_num_parents:
        nodes_with_access_parents = list(map(lambda x: x[0], (filter(lambda x: x[1] > 2, graph.out_degree()))))
        edges_to_remove = np.vstack(np.array(list(
            map(lambda child: np.array(list(map(lambda succ: (child, succ), list(graph.successors(child))))),
                nodes_with_access_parents))))
        graph.remove_edges_from(edges_to_remove)
        graph.remove_nodes_from(list(nx.isolates(graph)))
    # TODO: consider having a list of elements to remove and only remove them after checking for violations.
    sort_order = np.array(list(nx.topological_sort(graph)))[::-1]
    return rel[sort_order][:, sort_order], sort_order


def get_ancestor_matrix(rel):
    """
    Get Ancestor Matrix
    :param rel: relationship csr_matrix
    :return:
    """
    am = rel
    temp_mat = rel
    while temp_mat.nnz > 0:
        temp_mat = temp_mat.dot(rel)
        am += temp_mat
    return am.astype(np.bool)


# each entry in the CAM matrix is 1 if the couple have a common ancestor.
def get_CAM(AM):
    n = AM.shape[0]
    sub = AM + eye(n)
    return sub.dot(sub.T).astype(np.bool)


# use the CAM matrix to see how many nonzero entries IBD has (much faster than computing IBD each time)
def count_IBD_nonzero(rel):
    return get_CAM(get_ancestor_matrix(rel)).nnz


def get_only_relevant_indices(rel, subset):
    AM = get_ancestor_matrix(rel)
    with_ancestors = np.unique(np.concatenate((subset,
                                               AM[subset].nonzero()[1])))

    # remove individuals that are not subset or ancestors to subset
    sub_rel = rel[with_ancestors][:, with_ancestors]
    sub_AM = AM[with_ancestors][:, with_ancestors]

    # remove those who are related to only 1 individual of the subset
    only_subset = np.in1d(with_ancestors, subset)
    sub_CAM = get_CAM(sub_AM)
    valid_nonsubset = sub_CAM[only_subset][:, ~only_subset].sum(axis=0).A1 > 1
    valid_nonsubset_indices = with_ancestors[np.where(~only_subset)[0][valid_nonsubset]]
    relevant_subset = np.unique(np.concatenate((subset, valid_nonsubset_indices)))
    return relevant_subset, np.searchsorted(relevant_subset, subset)


# indices should be sorted
def organize_rel(rel, subset=None, remove_cycles=False, check_num_parents=False):
    rel, topo_order = topo_sort(rel, remove_cycles=remove_cycles, check_num_parents=check_num_parents)
    topo_order_argsort = np.argsort(topo_order)
    if subset is not None:
        # TODO: check this again. do it with unique ids
        subset_in_topo_order = topo_order_argsort[subset]
        relevant_subset, subset_in_relevant_subset = \
            get_only_relevant_indices(rel, subset_in_topo_order)
        return rel[relevant_subset][:, relevant_subset], subset_in_relevant_subset
    return rel, topo_order


if __name__ == "__main__":
    from ..Examples.GraphExamples import henderson_example, cam_example
    from ..Matrices.Numerator import simple_numerator
    from ..Simulation.Pedigree import simulate_tree

    rel, _, _, _ = henderson_example()
    print(topo_sort(rel))

    rel, subset = cam_example()
    print(get_only_relevant_indices(rel, subset))

    rev_rel = rel[::-1][:, ::-1]
    rev_subset = np.array([4, 3])
    relevant_rel, subset_indices = organize_rel(rev_rel, rev_subset)

    sub_ibd1 = simple_numerator(relevant_rel)[0][subset_indices][:, subset_indices]

    relevant_rel, subset_indices = organize_rel(rel, subset)
    sub_ibd2 = simple_numerator(relevant_rel)[0][subset_indices][:, subset_indices]

    rel, _, _ = simulate_tree(10000, 0.001, 1.4, 0.9)
    subset = np.sort(np.random.choice(np.arange(10000), size=500, replace=False))

    relevant_subset, _ = get_only_relevant_indices(rel, subset)

    relevant_ibd, _, _ = simple_numerator(rel[relevant_subset][:, relevant_subset])
    subset_inside_relevant = np.in1d(relevant_subset, subset)
    ibd1 = relevant_ibd[subset_inside_relevant][:, subset_inside_relevant]

    full_ibd, _, _ = simple_numerator(rel)
    ibd2 = full_ibd[subset][:, subset]

    assert (ibd1 - ibd2).nnz == 0

    shuffle = np.arange(10000)
    np.random.shuffle(shuffle)
    shf_rel = rel[shuffle][:, shuffle]
    shf_subset = np.argsort(shuffle)[subset]

    shf_relevant_rel, shf_subset_indices = organize_rel(shf_rel, shf_subset)

    sub_rel1 = rel[subset][:, subset]
    sub_rel2 = shf_relevant_rel[shf_subset_indices][:, shf_subset_indices]

    assert (sub_rel1 - sub_rel2).nnz == 0

    sub_ibd1 = simple_numerator(rel)[0][subset][:, subset]
    sub_ibd2 = simple_numerator(shf_relevant_rel)[0][shf_subset_indices][:, shf_subset_indices]

    sub_ibd3 = simple_numerator(organize_rel(rel, subset)[0])[0][organize_rel(rel, subset)[1]][:,
               organize_rel(rel, subset)[1]]

    assert (sub_ibd1 - sub_ibd3).nnz == 0

    assert (sub_ibd1 - sub_ibd2).nnz == 0
