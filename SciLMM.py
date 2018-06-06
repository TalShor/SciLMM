from FileFormats.FAM import read_fam, write_fam
from Simulation.Pedigree import simulate_tree
from Matrices.Numerator import simple_numerator
from Matrices.Dominance import dominance
from Matrices.Epistasis import pairwise_epistasis
from Matrices.Relationship import organize_rel
import os
from Matrices.SparseMatrixFunctions import load_sparse_csr, save_sparse_csr
from Estimation.HE import compute_HE
from Estimation.LMM import LMM ,SparseCholesky
import numpy as np
import argparse
from Simulation.Phenotype import simulate_phenotype

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SciLMM')

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
                             'or if only a subset is of interest the\'re phenotype will contain 1')

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

    if args.ibd or args.epis or args.dom:
        if not os.path.exists(args.output_folder):
            raise Exception("The output folder does not exists")

    if args.he or args.lmm:
        if args.y is None and args.gen_y is False:
            raise Exception("Can't estimate without a target value (--y)")

    rel, interest_in_relevant = None, None
    if args.fam:
        rel_org, sex, interest, entries_dict = read_fam(args.fam)
        rel, interest_in_relevant = organize_rel(rel_org, interest)
        # TODO: have to do sex as well in this version
        entries_list = np.array(list(entries_dict.values()))[interest_in_relevant]
        np.save(os.path.join(args.output_folder, "entires_ids.npy"), entries_list)
    elif args.simulate:
        if args.sample_size <= 0:
            raise Exception("Sample size should be a positive number")
        if (args.sparsity_factor <= 0) or (args.sparsity_factor >= 1):
            raise Exception("Sparsity factor is within the range (0, 1)")
        if args.gen_exp <= 0:
            raise Exception("gen_exp is a positive number")
        if (args.init_keep_rate <= 0) or (args.init_keep_rate > 1):
            raise Exception("init_keep_rate is within the range (0, 1)")
        rel, sex, _ = simulate_tree(args.sample_size, args.sparsity_factor,
                                    args.gen_exp, args.init_keep_rate)
        write_fam(os.path.join(args.output_folder, "rel.fam"), rel, sex, None)
   
    # if no subset of interest has been specified, keep all indices
    if interest_in_relevant is None:
        interest_in_relevant = np.ones((rel.shape[0])).astype(np.bool)
    ibd, epis, dom = None, None, None
    if args.ibd_path:
        ibd = load_sparse_csr(os.path.join(args.output_folder, "IBD.npz"))
    elif args.ibd:
        if rel is None:
            raise Exception("No relationship matrix given")
        ibd, L, D = simple_numerator(rel)
        # keep the original L and D because they are useless otherwise 
        save_sparse_csr(os.path.join(args.output_folder, "IBD.npz"), ibd)
        save_sparse_csr(os.path.join(args.output_folder, "L.npz"), L)
        save_sparse_csr(os.path.join(args.output_folder, "D.npz"), D)

    if args.epis_path:
        epis = load_sparse_csr(os.path.join(args.output_folder, "Epistasis.npz"))
    elif args.epis:
        if ibd is None:
            raise Exception("Pairwise-epistasis requires an ibd matrix")
        epis = pairwise_epistasis(ibd)
        save_sparse_csr(os.path.join(args.output_folder, "Epistasis.npz"), epis)

    if args.dom_path:
        dom = load_sparse_csr(os.path.join(args.output_folder, "Dominance.npz"))
    elif args.dom:
        if ibd is None or rel is None:
            raise Exception("Dominance requires both an ibd matrix and a relationship matrix")
        dom = dominance(rel, ibd)
        save_sparse_csr(os.path.join(args.output_folder, "Dominance.npz"), dom)

    covariance_matrices = []
    for mat in [ibd, epis, dom]:
        if mat is not None:
            covariance_matrices.append(mat)

    cov = sex[:, np.newaxis]
    if args.cov is not None:
        cov = np.hstack((cov, np.load(args.cov)))

    y = None
    if args.gen_y:
        sigs = np.random.rand(len(covariance_matrices) + 1); sigs /= sigs.sum()
        fe = np.random.rand(cov.shape[1] + args.intercept) / 100
        print("Generating y with fixed effects: {} and sigmas : {}".format(fe, sigs))
        y = simulate_phenotype(covariance_matrices, cov, sigs, fe, args.intercept)
        np.save(os.path.join(args.output_folder, "y.npy"), y)
    if args.y is not None:
        y = np.load(args.y)

    if args.he:
        print(compute_HE(y, cov, covariance_matrices, args.intercept))

    if args.lmm:
        print(LMM(SparseCholesky(), covariance_matrices, cov, y,
                  with_intercept=args.intercept, reml=args.reml, sim_num=args.sim_num))
