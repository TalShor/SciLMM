from FileFormats.FAM import translate_fam
from Simulation.Pedigree import simulate_tree
from Matrices.Numerator import simple_numerator
from Matrices.Dominance import dominance
from Matrices.Epistasis import pairwise_epistasis
import os
from Matrices.SparseMatrixFunctions import load_sparse_csr, save_sparse_csr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SciLMM')

    # simulation values
    parser.add_argument('simulate', dest='simulate', action='store_true', default=False,
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

    parser.add_argument('--IBD_path', dest='ibd_path', type=str, default=None,
                        help='location of the .npz file for the IBD matrix ' + \
                             '(if you already had build the matrix via the --IBD option)')
    parser.add_argument('--Epis_path', dest='epis_path', type=str, default=None,
                        help='location of the .npz file for the Epistasis matrix ' + \
                             '(if you already had build the matrix via the --Epistasis option)')
    parser.add_argument('--Dom_path', dest='dom_path', type=str, default=None,
                        help='location of the .npz file for the Dominance matrix ' + \
                             '(if you already had build the matrix via the --Dominance option)')

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

    parser.add_argument('--output_folder', dest='output_folder', type=str, default='.',
                        help='which folder it should save the output to.')

    args = parser.parse_args()

    if args.ibd or args.epis or args.dom:
        if not os.path.exists(args.output_folder):
            raise Exception("The output folder does not exists")

    rel = None
    if args.fam:
        rel, sex, interest = translate_fam(args.fam)
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

    ibd, epis, dom = None
    if args.ibd_path:
        ibd = load_sparse_csr(os.path.join(args.output_folder, "IBD.npz"))
    elif args.ibd:
        if rel is None:
            raise Exception("No relationship matrix given")
        ibd = simple_numerator(rel)
        save_sparse_csr(os.path.join(args.output_folder, "IBD.npz"), ibd)

    if args.epis_path:
        epis = load_sparse_csr(os.path.join(args.output_folder, "Epistasis.npz"))
    elif args.epis:
        if ibd is None:
            raise Exception("Pairwise-epistasis requires an ibd matrix")
        epis = pairwise_epistasis(ibd)
        save_sparse_csr(os.path.join(args.output_folder, "Epis.npz"), epis)

    if args.dom_path:
        dom = load_sparse_csr(os.path.join(args.output_folder, "Dominance.npz"))
    elif args.dom:
        if ibd is None or rel is None:
            raise Exception("Dominance requires both an ibd matrix and a relationship matrix")
        dom = dominance(rel, ibd)
        save_sparse_csr(os.path.join(args.output_folder, "Dominance.npz"), dom)


# indices should be sorted
def get_sub_rel(rel, indices):


# TODO: implement .fam thing...
"""
.fam (PLINK sample information file)
Sample information file accompanying a .bed binary genotype table. (--make-just-fam can be used to update just this file.) Also generated by '--recode lgen' and '--recode rlist'.

A text file with no header line, and one line per sample with the following six fields:

Family ID ('FID')
Within-family ID ('IID'; cannot be '0')
Within-family ID of father ('0' if father isn't in dataset)
Within-family ID of mother ('0' if mother isn't in dataset)
Sex code ('1' = male, '2' = female, '0' = unknown)
Phenotype value ('1' = control, '2' = case, '-9'/'0'/non-numeric = missing data if case/control)

"""