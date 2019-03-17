import os
from unittest import TestCase

import numpy as np

from scilmm.Estimation.HE import compute_HE
from scilmm.Estimation.LMM import LMM, SparseCholesky
from scilmm.FileFormats.FAM import read_fam, write_fam
from scilmm.FileFormats.pedigree import Pedigree
from scilmm.Matrices.Dominance import dominance
from scilmm.Matrices.Epistasis import pairwise_epistasis
from scilmm.Matrices.Numerator import simple_numerator
from scilmm.Matrices.Relationship import organize_rel
from scilmm.Matrices.SparseMatrixFunctions import load_sparse_csr, save_sparse_csr
from scilmm.IBDComputeWrapper import scilmm_parse_arguments
from scilmm.Simulation.Pedigree import simulate_tree
from scilmm.Simulation.Phenotype import simulate_phenotype


class IBDCompute:
    def __init__(self, output_folder='.', simulate=False, **kwargs):
        """
        Initiating an class IBDCompute class and computing IBD.
        :param output_folder: Path for output files
        """
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.kwargs = kwargs
        if simulate:
            self.simulate_relationship(**kwargs)
            self.compute_ibd()

    def compute_relationships(self, pedigree_file_path=None):
        """
        Compute a relationship matrix, and store in self.pedigree.
        Notice that self.pedigree is of type FileFormats.pedigree and can also be loaded in a different method.
        :param pedigree_file_path: path of FAM file.
        :return: A class Pedigree object, and an entries list.
        """
        if pedigree_file_path is None:
            assert hasattr(self, 'pedigree'), "In order to compute relationship must first load pedigree either " \
                                              "through a variable to .compute_relationships or through .load_pedigree "
        else:
            self.pedigree = Pedigree()
            self.pedigree.load_pedigree(pedigree_file_path)
            self.pedigree.compute_all_values()
        entries_list = np.array(list(self.pedigree.entries_dict.values()))[self.pedigree.interest]
        np.save(os.path.join(self.output_folder, "entries_ids.npy"), entries_list)
        return self.pedigree, entries_list

    def simulate_relationship(self, sample_size=100000, sparsity_factor=0.001, gen_exp=1.4, init_keep_rate=0.8, **kwargs):
        """
        Create a simulated example of a pedigree and compute its relationship.
        :param sample_size: Size of the cohort.
        :param sparsity_factor: Number of nonzero entries in the IBD matrix.
        :param gen_exp: Gen size = gen_exp X prev gen size.
        :param init_keep_rate: 1 - number of edges to remove before iteration begins.
        :return: A class Pedigree object, and an entry list of the simulation.
        """
        assert sample_size>0, "Sample size should be a positive number"
        assert(sparsity_factor > 0) and (sparsity_factor < 1), \
            "Sparsity factor must be within the range (0, 1)"
        assert gen_exp>0, "gen_exp should be a positive number"
        assert (init_keep_rate > 0) and (init_keep_rate < 1), \
            "init_keep_rate must be within the range (0, 1)"
        rel, sex, _ = simulate_tree(sample_size, sparsity_factor,
                                    gen_exp, init_keep_rate)
        write_fam(os.path.join(self.output_folder, "rel.fam"), rel, sex, None)
        return self.compute_relationships(os.path.join(self.output_folder, "rel.fam"))

    def compute_ibd(self):
        _, entries_list = self.compute_relationships()
        ibd, L, D = simple_numerator(self.pedigree.relationship)
        # keep the original L and D because they are useless otherwise
        save_sparse_csr(os.path.join(self.output_folder, "IBD.npz"), ibd)
        save_sparse_csr(os.path.join(self.output_folder, "L.npz"), L)
        save_sparse_csr(os.path.join(self.output_folder, "D.npz"), D)

##########
# This should all go in a different file that calculates the SciLMM
# def SciLMM_foo(rel, ibd=False, epis=False, dom=False, epis_path=False,
#                dom_path=False, gen_y=False, y=None, cov=None, he=False, lmm=False, reml=False, sim_num=100,
#                intercept=False, output_folder='.'):
#     if he or lmm:
#         if y is None and gen_y is False:
#             raise Exception("Can't estimate without a target value (--y)")
#
#     if epis_path:
#         epis = load_sparse_csr(os.path.join(output_folder, "Epistasis.npz"))
#     elif epis:
#         if ibd is None:
#             raise Exception("Pairwise-epistasis requires an ibd matrix")
#         epis = pairwise_epistasis(ibd)
#         save_sparse_csr(os.path.join(output_folder, "Epistasis.npz"), epis)
#     else:
#         epis = None
#
#     if dom_path:
#         dom = load_sparse_csr(os.path.join(output_folder, "Dominance.npz"))
#     elif dom:
#         if ibd is None or rel is None:
#             raise Exception("Dominance requires both an ibd matrix and a relationship matrix")
#         dom = dominance(rel, ibd)
#         save_sparse_csr(os.path.join(output_folder, "Dominance.npz"), dom)
#     else:
#         dom = None
#
#     covariance_matrices = []
#     for mat in [ibd, epis, dom]:
#         if mat is not None:
#             covariance_matrices.append(mat)
#
#     if cov is not None:
#         cov = np.hstack(np.load(cov))
#     else:
#         cov = sex[:, np.newaxis]
#
#     y = None
#     if gen_y:
#         sigs = np.random.rand(len(covariance_matrices) + 1);
#         sigs /= sigs.sum()
#         fe = np.random.rand(cov.shape[1] + intercept) / 100
#         print("Generating y with fixed effects: {} and sigmas : {}".format(fe, sigs))
#         y = simulate_phenotype(covariance_matrices, cov, sigs, fe, intercept)
#         np.save(os.path.join(output_folder, "y.npy"), y)
#     if y is not None:
#         y = np.load(y)
#
#     if he:
#         print(compute_HE(y, cov, covariance_matrices, intercept))
#
#     if lmm:
#         print(LMM(SparseCholesky(), covariance_matrices, cov, y,
#                   with_intercept=intercept, reml=reml, sim_num=sim_num))


if __name__ == "__main__":
    args = scilmm_parse_arguments()
    ibd_creator = IBDCompute(**args.__dict__)
