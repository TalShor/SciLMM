import argparse


def scilmm_parse_arguments():
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
                             'or if only a subset is of interest their phenotype will contain 1')

    parser.add_argument('--remove_cycles', dest='remove_cycles', action='store_true', default=False,
                        help='Remove cycles from relationship matrix.' +
                             'WARNING: there should no be any cycles. All nodes in cycles will be removed.')

    parser.add_argument('--remove_access_parents', dest='check_num_parents', action='store_true', default=False,
                        help='Remove relations of nodes with too many parents.' +
                             'WARNING: All individuals should have no more than 2 parents.' +
                             'Access edges will be removed, not nodes.')
    #
    # parser.add_argument('--IBD', dest='ibd', action='store_true', default=False,
    #                     help='Create IBD matrix')
    # parser.add_argument('--Epistasis', dest='epis', action='store_true', default=False,
    #                     help='Create pairwise-epistasis matrix')
    # parser.add_argument('--Dominance', dest='dom', action='store_true', default=False,
    #                     help='Create dominance matrix')
    #
    # parser.add_argument('--IBD_exists', dest='ibd_path', action='store_true', default=False,
    #                     help='existence of the .npz file for the IBD matrix ' +
    #                          '(if you already had build the matrix via the --IBD option)')
    # parser.add_argument('--Epis_exists', dest='epis_path', action='store_true', default=False,
    #                     help='existence of the .npz file for the Epistasis matrix ' +
    #                          '(if you already had build the matrix via the --Epistasis option)')
    # parser.add_argument('--Dom_exists', dest='dom_path', action='store_true', default=False,
    #                     help='existence of the .npz file for the Dominance matrix ' + \
    #                          '(if you already had build the matrix via the --Dominance option)')
    #
    # parser.add_argument('--generate_y', dest='gen_y', action='store_true', default=False,
    #                     help='Generate a random y')
    #
    # parser.add_argument('--y', dest='y', type=str, default=None,
    #                     help='the phenotype (npy file containing an n sized numpy array)')
    # parser.add_argument('--covariates', dest='cov', type=str, default=None,
    #                     help='the covaraites, not including sex (npy file containing an nxc sized numpy array)')
    #
    # parser.add_argument('--HE', dest='he', action='store_true', default=False,
    #                     help='Estimate fixed effects and covariance coefficients via Haseman-Elston')
    # parser.add_argument('--LMM', dest='lmm', action='store_true', default=False,
    #                     help='Estimate fixed effects and covariance coefficients via Linear mixed models')
    # parser.add_argument('--REML', dest='reml', action='store_true', default=False,
    #                     help='Use REML instead of simple maximum likelihood')
    # parser.add_argument('--sim_num', dest='sim_num', type=int, default=100,
    #                     help='Number of simulated vectors')
    # parser.add_argument('--fit_intercept', dest='intercept', action='store_true', default=False,
    #                     help='Use an intercept as a covariate')
    #
    # parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
    #                     help='prints more information along the run.')
    parser.add_argument('--output_folder', dest='output_folder', type=str, default='.',
                        help='which folder it should save the output to.')

    args = parser.parse_args()

    return args
