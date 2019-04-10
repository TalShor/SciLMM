import argparse
import os

import numpy as np

from scilmm.Estimation.HE import compute_HE
from scilmm.Estimation.LMM import LMM, SparseCholesky
from scilmm.FileFormats.FAM import read_fam, write_fam
from scilmm.Matrices.Dominance import dominance
from scilmm.Matrices.Epistasis import pairwise_epistasis
from scilmm.Matrices.Numerator import simple_numerator
from scilmm.Matrices.Relationship import organize_rel
from scilmm.Matrices.SparseMatrixFunctions import load_sparse_csr, save_sparse_csr
from scilmm.Simulation.Pedigree import simulate_tree
from scilmm.Simulation.Phenotype import simulate_phenotype


def parse_arguments():
    parser = argparse.ArgumentParser(description='scilmm')

    # simulation values
    parser.add_argument('--simulate', dest='simulate', action='store_true', default=False,
                        help='Run simulations')
    parser.add_argument('--sample_size', dest='sample_size', type=int, default=100000,
                        help='Size of the cohort')
    parser.add_argument('--sparsity_factor', dest='sparsity_factor', type=float, default=0.001,
                        help='Number of nonzero entries in the IBD matrix')
    parser.add_argument('--gen_exp', dest='gen_exp', type=float, default=1.4,
                        help='Gen size = gen_exp X prev gen size')
    parser.add_argument('--init_keep_rate', dest='init_keep_rate', type=float, default=0.8,
                        help='1 - number of edges to remove before iteration begins')

    parser.add_argument('--fam', dest='fam', type=str, default=None,
                        help='.fam file representing the pedigree. ' +
                             'the phenotype column contains all 0 if everyone is of interest, ' +
                             'or if only a subset is of interest their phenotype will contain 1')

    parser.add_argument('--remove_cycles', dest='remove_cycles', action='store_true', default=False,
                        help='Remove cycles from relationship matrix.' +
                             'WARNING: there should no be any cycles. All nodes in cycles will be removed.')

    parser.add_argument('--remove_access_parents', dest='check_num_parents', action='store_true', default=False,
                        help='Remove relations of nodes with too many parents.' +
                             'WARNING: All individuals should have no more than 2 parents.' +
                             'Access edges will be removed, not nodes.')

    parser.add_argument('--IBD', dest='ibd', action='store_true', default=False,
                        help='Create IBD matrix')
    parser.add_argument('--Epistasis', dest='epis', action='store_true', default=False,
                        help='Create pairwise-epistasis matrix')
    parser.add_argument('--Dominance', dest='dom', action='store_true', default=False,
                        help='Create dominance matrix')

    parser.add_argument('--IBD_exists', dest='ibd_path', action='store_true', default=False,
                        help='existence of the .npz file for the IBD matrix ' + \
                             '(if you already had build the matrix via the --IBD option)')
    parser.add_argument('--Epis_exists', dest='epis_path', action='store_true', default=False,
                        help='existence of the .npz file for the Epistasis matrix ' + \
                             '(if you already had build the matrix via the --Epistasis option)')
    parser.add_argument('--Dom_exists', dest='dom_path', action='store_true', default=False,
                        help='existence of the .npz file for the Dominance matrix ' + \
                             '(if you already had build the matrix via the --Dominance option)')

    parser.add_argument('--generate_y', dest='gen_y', action='store_true', default=False,
                        help='Generate a random y')

    parser.add_argument('--y', dest='y', type=str, default=None,
                        help='the phenotype (npy file containing an n sized numpy array)')
    parser.add_argument('--covariates', dest='cov', type=str, default=None,
                        help='the covaraites, not including sex (npy file containing an nxc sized numpy array)')

    parser.add_argument('--HE', dest='he', action='store_true', default=False,
                        help='Estimate fixed effects and covariance coefficients via Haseman-Elston')
    parser.add_argument('--LMM', dest='lmm', action='store_true', default=False,
                        help='Estimate fixed effects and covariance coefficients via Linear mixed models')
    parser.add_argument('--REML', dest='reml', action='store_true', default=False,
                        help='Use REML instead of simple maximum likelihood')
    parser.add_argument('--sim_num', dest='sim_num', type=int, default=100,
                        help='Number of simulated vectors')
    parser.add_argument('--fit_intercept', dest='intercept', action='store_true', default=False,
                        help='Use an intercept as a covariate')

    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='prints more information along the run.')
    parser.add_argument('--output_folder', dest='output_folder', type=str, default='.',
                        help='which folder it should save the output to.')

    args = parser.parse_args()

    return args


def SciLMM(simulate=False, sample_size=100000, sparsity_factor=0.001, gen_exp=1.4, init_keep_rate=0.8, fam=None,
           ibd=False, epis=False, dom=False, ibd_path=False, epis_path=False,
           dom_path=False, gen_y=False, y=None, cov=None, he=False, lmm=False, reml=False, sim_num=100, intercept=False,
           verbose=False, output_folder='.', remove_cycles=False, check_num_parents=False):
    if ibd or epis or dom:
        if not os.path.exists(output_folder):
            raise Exception("The output folder does not exists")

    if he or lmm:
        if y is None and gen_y is False:
            raise Exception("Can't estimate without a target value (--y)")

    rel, interest_in_relevant = None, None
    if fam:
        rel_org, sex, interest, entries_dict = read_fam(fam_file_path=fam)
        rel, interest_in_relevant = organize_rel(rel_org, interest, remove_cycles=remove_cycles,
                                                 check_num_parents=check_num_parents)
        # TODO: have to do sex as well in this version
        entries_list = np.array(list(entries_dict.values()))[interest_in_relevant]
        np.save(os.path.join(output_folder, "entries_ids.npy"), entries_list)
    elif simulate:
        if sample_size <= 0:
            raise Exception("Sample size should be a positive number")
        if (sparsity_factor <= 0) or (sparsity_factor >= 1):
            raise Exception("Sparsity factor is within the range (0, 1)")
        if gen_exp <= 0:
            raise Exception("gen_exp is a positive number")
        if (init_keep_rate <= 0) or (init_keep_rate > 1):
            raise Exception("init_keep_rate is within the range (0, 1)")
        rel, sex, _ = simulate_tree(sample_size, sparsity_factor,
                                    gen_exp, init_keep_rate)
        write_fam(os.path.join(output_folder, "rel.fam"), rel, sex, None)

    # if no subset of interest has been specified, keep all indices
    if interest_in_relevant is None:
        interest_in_relevant = np.ones((rel.shape[0])).astype(np.bool)

    if ibd_path:
        ibd = load_sparse_csr(os.path.join(output_folder, "IBD.npz"))
    elif ibd:
        if rel is None:
            raise Exception("No relationship matrix given")
        ibd, L, D = simple_numerator(rel)
        # keep the original L and D because they are useless otherwise 
        save_sparse_csr(os.path.join(output_folder, "IBD.npz"), ibd)
        save_sparse_csr(os.path.join(output_folder, "L.npz"), L)
        save_sparse_csr(os.path.join(output_folder, "D.npz"), D)
    else:
        ibd = None

    if epis_path:
        epis = load_sparse_csr(os.path.join(output_folder, "Epistasis.npz"))
    elif epis:
        if ibd is None:
            raise Exception("Pairwise-epistasis requires an ibd matrix")
        epis = pairwise_epistasis(ibd)
        save_sparse_csr(os.path.join(output_folder, "Epistasis.npz"), epis)
    else:
        epis = None

    if dom_path:
        dom = load_sparse_csr(os.path.join(output_folder, "Dominance.npz"))
    elif dom:
        if ibd is None or rel is None:
            raise Exception("Dominance requires both an ibd matrix and a relationship matrix")
        dom = dominance(rel, ibd)
        save_sparse_csr(os.path.join(output_folder, "Dominance.npz"), dom)
    else:
        dom = None

    covariance_matrices = []
    for mat in [ibd, epis, dom]:
        if mat is not None:
            covariance_matrices.append(mat)

    if cov is not None:
        cov = np.hstack((cov, np.load(cov)))
    else:
        cov = sex[:, np.newaxis]

    y = None
    if gen_y:
        sigs = np.random.rand(len(covariance_matrices) + 1);
        sigs /= sigs.sum()
        fe = np.random.rand(cov.shape[1] + intercept) / 100
        print("Generating y with fixed effects: {} and sigmas : {}".format(fe, sigs))
        y = simulate_phenotype(covariance_matrices, cov, sigs, fe, intercept)
        np.save(os.path.join(output_folder, "y.npy"), y)
    if y is not None:
        y = np.load(y)

    if he:
        print(compute_HE(y, cov, covariance_matrices, intercept))

    if lmm:
        print(LMM(SparseCholesky(), covariance_matrices, cov, y,
                  with_intercept=intercept, reml=reml, sim_num=sim_num))


if __name__ == "__main__":
    args = parse_arguments()
    SciLMM(**args.__dict__)
