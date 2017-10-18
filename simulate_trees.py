import numpy as np
from itertools import combinations, product
from scipy.sparse import lil_matrix, eye
from scipy.sparse.csgraph import connected_components
from SideAlgos.SparseFunctions import *


def create_possible_parents(gen_ids, gens, sex_gen,
                            same_generation_const, diff_generation_const,
                            cg):

    cg_same_gen_const = int(same_generation_const * gens[cg + 1])
    same_gen = np.vstack((
        np.random.choice(gen_ids[cg + 1][sex_gen[cg + 1] == 0], cg_same_gen_const),
        np.random.choice(gen_ids[cg + 1][sex_gen[cg + 1] == 1], cg_same_gen_const))).T

    single_parents = np.array([[i, -1] for i in gen_ids[cg + 1]])
    no_parents = np.array([[-1, -1]])

    all_parents = [same_gen, single_parents, no_parents]

    if cg < len(gens) - 2:
        cg_diff_const = int(diff_generation_const * gens[cg + 1])
        diff_gen_fm = np.vstack((
            np.random.choice(gen_ids[cg + 1][sex_gen[cg + 1] == 0], cg_diff_const),
            np.random.choice(gen_ids[cg + 2][sex_gen[cg + 2] == 1], cg_diff_const))).T

        diff_gem_mf = np.vstack((
            np.random.choice(gen_ids[cg + 1][sex_gen[cg + 1] == 1], cg_diff_const),
            np.random.choice(gen_ids[cg + 2][sex_gen[cg + 2] == 0], cg_diff_const))).T

        diff_gen = np.concatenate((diff_gem_mf, diff_gen_fm))
        all_parents.insert(1, diff_gen)

    return np.concatenate(all_parents), np.array([p.shape[0] for p in all_parents])


def norm_possibilities(parents_probs, num_parents_in_type, cg, gens, probs = None):
    if cg == len(gens) - 2:
        parents_probs = parents_probs[[0, 2, 3]]
        parents_probs /= parents_probs.sum()
    if probs is None:
        probs = np.ones((sum(num_parents_in_type)))
    for i in range(len(num_parents_in_type)):
        s, e = sum(num_parents_in_type[:i]), sum(num_parents_in_type[:i+1])
        probs[s:e] = (parents_probs[i] * probs[s:e]) / probs[s:e].sum()

    return probs


def get_index_of_parents_couples(all_parents, n):
    same_gen_couples = [list() for _ in range(n)]
    for j, couple in enumerate(all_parents):
        if couple[0] != -1:
            same_gen_couples[couple[0]].append([couple, j])
        if couple[1] != -1:
            same_gen_couples[couple[1]].append([couple, j])

    return np.array(same_gen_couples)


def simulate_tree(ind_count, exp_rate, gen_count, name, parents_probs_const, children_probs, same_generation_const, diff_generation_const):
    exp_rate_list = exp_rate ** np.arange(gen_count)
    x = ind_count / float((exp_rate_list).sum())
    gens = (x * exp_rate_list).astype(np.int)[::-1]
    n = gens.sum()
    sex = np.random.randint(0, 2, size=(n))
    sex_gen = [sex[gens[0:i].sum():gens[0:i+1].sum()] for i, gen in enumerate(gens)]
    gen_ids = [np.arange(gens[0:i].sum(),gens[0:i+1].sum()) for i, gen in enumerate(gens)]


    # both parents, cross generation, single parent, no parents
    parents = np.zeros((n, 2)) - 1
    parents = parents.astype(np.int)
    # need to do the math
    num_child_const = 3

    count_parents = np.zeros((n + 1))
    # last one has no parents - and the one before that has no mix
    for cg in range(len(gens) - 1):

        all_parents, num_parents_in_type = \
            create_possible_parents(gen_ids, gens, sex_gen,
                same_generation_const, diff_generation_const, cg)

        all_probs = norm_possibilities(parents_probs_const, num_parents_in_type, cg, gens)

        same_gen_couples = get_index_of_parents_couples(all_parents, n)

        for ind in gen_ids[cg]:
            while True:

		all_probs /= all_probs.sum()
                p = all_parents[np.random.choice(np.arange(all_probs.shape[0]), p = all_probs)]
                # no parents
                if p[0] == -1:
                    num_of_childrens = 0
                else:
                    num_of_childrens = count_parents[p[0]] if p[1] == -1 else int(max(count_parents[p[0]], count_parents[p[1]]))
                if num_of_childrens < children_probs.shape[0]:
                    next_child_prob = children_probs[num_of_childrens]
                    count_parents[p[0]] += 1
                    count_parents[p[1]] += 1
                    break

            if p[0] == -1:
                pass
            else:
                if p[1] == -1:
                    p_inside = np.array(same_gen_couples[p[0]])
                else:
                    p_inside = np.concatenate((same_gen_couples[p[0]], same_gen_couples[p[1]]))
                    p_inside = p_inside[np.unique(p_inside[:, 1], return_index=True)[1]]
                p_inside_only_couples = np.array(p_inside[:, 0].tolist()) #  np.array([el[0] for el in p_inside])                
                p_ind = np.where((p_inside_only_couples[:, 0] == p[0]) & (p_inside_only_couples[:, 1] == p[1]))[0]
                p_j = p_inside[p_ind, 1][0]
                all_probs[p_inside[:, 1].astype(np.int)] *= 0.5
                all_probs[p_j] *= 2 * next_child_prob

                all_probs = norm_possibilities(parents_probs_const, num_parents_in_type, cg, gens, all_probs)

            parents[ind] = p


    #print "Done simulating"

    rel = lil_matrix((n, n), dtype=np.bool)
    for i in range(2):
        p = np.vstack((np.arange(n), parents[:, i]))
        p = p[:, p[1] != -1]
        rel[p[0], p[1]] = 1

    rel = rel.tocsr().astype(np.bool)
    #save_sparse_csr("DataSets/yaniv/simulated_tree_500000_1.6_4.npz", rel)
    save_sparse_csr(name + ".npz", rel)
    np.save(name + ".npy", count_parents)
    np.save(name + "_sex.npy", sex)


    #print rel.sum(axis=0).max(), rel.sum(axis=1).max()

    #a =rel.sum(axis=1)
    #print np.count_nonzero(a == 0), np.count_nonzero(a ==1 ), np.count_nonzero(a == 2)
    #print np.unique(np.array(rel.sum(axis=0))[0], return_counts = True)

    #M = rel + rel.T
    #_, cc = connected_components(M + M.T, False)
    #print np.sort(np.unique(cc, return_counts=True)[1])[::-1][0:20]


    #am = rel.copy()
    #rel_copy = rel.copy()
    #while rel_copy.nnz > 0:
    #    rel_copy = rel_copy.dot(rel)
    #    am = am + rel_copy



    #i_n = eye(am.shape[0])
    #cam = (am + i_n).dot((am + i_n).transpose(copy=True))
    #print "nonzero relationships : ", cam.nnz


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute sigams')
    parser.add_argument('--tree_size', dest='tree_size', type=int, default=1000)
    parser.add_argument('--exp_rate', dest='exp_rate', type=float, default=2.0)
    parser.add_argument('--gens', dest='gens', type=int, default=4)
    parser.add_argument('--name', dest='name')
    parser.add_argument('--parents_probs', dest='pp', nargs='+', default=[0.2, 0.1, 0.2, 0.5])
    parser.add_argument('--children_probs', dest='cp', nargs='+', default=[5, 0.1, 0.01, 0.001, 0.001, 0.001])



    args = parser.parse_args()
    tree_size, exp_rate, gens, name = args.tree_size, args.exp_rate, args.gens, args.name

    pp, cp = np.array(args.pp).astype(np.float64), np.array(args.cp).astype(np.float64)

    same_generation_const = 0.2
    diff_generation_const = 0.4


    simulate_tree(tree_size, exp_rate, gens, name, pp, cp, same_generation_const, diff_generation_const)

