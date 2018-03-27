import numpy as np
import itertools
from scipy.sparse import csr_matrix, triu, eye
from scipy.optimize import fmin
import argparse
from SideAlgos.SparseFunctions import save_sparse_csr
import os


# number of individuals per generation so that each generation is gen_exp bigger than the previous
def generation_size(sample_size, gen_exp):
    gens = np.array(map(lambda x: int(2 * (gen_exp ** x)), range(int(np.log(sample_size * gen_exp)/np.log(gen_exp)))))
    enough_gens = np.where(np.cumsum(gens) > sample_size)[0][0]
    return gens[:enough_gens].tolist() + [sample_size - gens[:enough_gens].sum()]


# create indices per generation
def get_generations(gen_sizes):
    start = 0
    gens = []
    for gen_size in gen_sizes:
        gens.append(np.arange(start, start + gen_size))
        start += gen_size
    return gens


# simulate households with respect to the United States Census Bureau official figs.
def households(generation, prev_generation):
    gen_size = generation.size
    single_parents = [[x] for x in np.random.choice(generation, int(gen_size * 0.32))]
    # first half will be sex1, second half is sex2
    same_generation = zip(np.random.choice(generation[:gen_size / 2], int(gen_size * 0.8 * 0.68)),
                          np.random.choice(generation[gen_size / 2:], int(gen_size * 0.8 * 0.68)))
    same_generation = [[x,y] for x,y in same_generation if x != y]
    if prev_generation == None:
        return np.array(single_parents + same_generation)

    # if not the top generation, we can mix between 2 generations
    prev_gen_size = prev_generation.size
    mix_generations = zip(np.random.choice(generation[:gen_size / 2], int(gen_size * 0.2 * 0.68 * 0.5)),
                          np.random.choice(prev_generation[prev_gen_size / 2:], int(gen_size * 0.2 * 0.68 * 0.5)))
    mix_generations += zip(np.random.choice(prev_generation[:prev_gen_size / 2], int(gen_size * 0.2 * 0.68 * 0.5)),
                          np.random.choice(generation[gen_size / 2:], int(gen_size * 0.2 * 0.68 * 0.5)))
    mix_generations = [[x, y] for x, y in mix_generations if x != y]
    return np.array(single_parents + same_generation + mix_generations)


# give a household for every individual. remove pairs to sparsify the matrix.
def combine_ind_to_households(generations, remove_rate):
    all_child_parent = []
    for i in range(1, len(generations)):
        hh = households(generations[i - 1], None if i == 1 else generations[i - 2])
        trios = zip(generations[i], hh[np.random.choice(hh.shape[0], generations[i].size)]) #hh[]
        child_parent = [[[child, parent] for parent in parents] for child, parents in trios]
        all_child_parent.append(list(itertools.chain(*child_parent)))
    all_child_parent = np.array(list(itertools.chain(*all_child_parent)))
    total_edges = all_child_parent.shape[0]
    return all_child_parent[np.random.choice(total_edges, int(total_edges * remove_rate))]


# ancestor matrix
def get_AM(rel):
    am = rel
    temp_mat = rel
    while temp_mat.nnz > 0:
        temp_mat = temp_mat.dot(rel)
        am += temp_mat
    return am


# use the CAM matrix to see how many nonzero entries IBD has (much faster than computing IBD each time)
def count_IBD_nonzero(rel):
    am = get_AM(rel)
    n = rel.shape[0]
    sub = am + eye(n)
    return sub.dot(sub.T).nnz


# return diff of number of nonzero ibd entries to wanted number given number of edges to remove
# 0 represents a matrix that has our wanted number of nonzero entries (up to 10%)
def count_with_removed_edges(edges_remove_part, rel_matrix, edges, wanted_number_of_ibds):
    num_edges = int(edges.shape[0] * edges_remove_part)
    sub_rel = rel_matrix.copy()
    sub_rel[edges[:num_edges, 0], edges[:num_edges, 1]] = False

    diff = np.abs(count_IBD_nonzero(sub_rel) - wanted_number_of_ibds)
    if diff < 0.1 * wanted_number_of_ibds:
        raise Exception(edges_remove_part)
    return diff


# find the right number of edges to remove via optimization process (hard stop when diff is 0)
def find_number_of_edges_to_remove(rel_matrix, edges, wanted_number_of_ibds):
    try:
        res = fmin(count_with_removed_edges, 0.3, args=(rel_matrix, edges, wanted_number_of_ibds,))
        if res > 0:
            return None
    except Exception as ex:
        return ex.args[0][0]


def simulate_tree(sample_size, sparse_factor, gen_exp, init_keep_rate):
    wanted_number_of_ibds = (sample_size ** 2) * sparse_factor

    # create an adjacency matrix representing
    gens = get_generations(generation_size(sample_size, gen_exp))
    edges = combine_ind_to_households(gens, init_keep_rate)
    rel_matrix = csr_matrix((np.ones(edges.shape[0]),
                             (edges[:, 0], edges[:, 1])),
                            shape=(sample_size, sample_size), dtype=np.bool)

    # check that every individual's parents are before him.
    assert triu(rel_matrix).nnz == 0

    # find the currect number of edges to remove till it matches
    np.random.shuffle(edges)
    res = find_number_of_edges_to_remove(rel_matrix, edges, wanted_number_of_ibds)
    if res == None:
        raise Exception("Did not find a good enough tree")
    else:
        num_edges = int(edges.shape[0] * res)
        rel_matrix[edges[:num_edges, 0], edges[:num_edges, 1]] = False
        rel_matrix.eliminate_zeros()

        # check that the resulting matrix is valid for our criteria
        assert np.abs(count_IBD_nonzero(rel_matrix) - wanted_number_of_ibds) < 0.1 * wanted_number_of_ibds
        sex = np.zeros((sample_size))
        gen_ind = np.zeros((sample_size))
        for i, gen in enumerate(gens):
            sex[gen[:gen.size / 2]] = 1
            gen_ind[gen] = i

        return rel_matrix, sex, gen_ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulate trees')
    parser.add_argument('--sample_size', dest='sample_size', type=int, default=100000,
                        help='Size of the cohort')
    parser.add_argument('--sparsity_factor', dest='sparsity_factor', type=float, default=0.001,
                        help='Number of nonzero entries in the IBD matrix')
    parser.add_argument('--gen_exp', dest='gen_exp', type=float, default=1.4,
                        help='Gen size = gen_exp X prev gen size')
    parser.add_argument('--init_keep_rate', dest='init_keep_rate', type=float, default=0.8,
                        help='1 - number of edges to remove before iteration begins')
    parser.add_argument('--save_folder', dest='save_folder', type=str, default='.',
                        help='which folder it should save the output to.')

    args = parser.parse_args()
    if args.sample_size <= 0:
        raise Exception("Sample size should be a positive number")
    if (args.sparsity_factor <= 0) or (args.sparsity_factor >= 1):
        raise Exception("Sparsity factor is within the range (0, 1)")
    if args.gen_exp <= 0:
        raise Exception("gen_exp is a positive number")
    if (args.init_keep_rate <= 0) or (args.init_keep_rate > 1):
        raise Exception("init_keep_rate is within the range (0, 1)")

    rel, sex, gen_ind = simulate_tree(args.sample_size, args.sparsity_factor, args.gen_exp, args.init_keep_rate)
    save_sparse_csr(os.path.join(args.save_folder, 'rel.npz'), rel)
    np.save(os.path.join(args.save_folder, 'sex.npy'), sex)
    np.save(os.path.join(args.save_folder, 'gen_ind.npy'), gen_ind)

