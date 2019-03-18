import os

import numpy as np

from scilmm.FileFormats.FAM import write_fam
from scilmm.FileFormats.pedigree import Pedigree
from scilmm.IBDComputeWrapper import scilmm_parse_arguments
from scilmm.Matrices.Numerator import simple_numerator
from scilmm.Matrices.SparseMatrixFunctions import save_sparse_csr
from scilmm.Simulation.Pedigree import simulate_tree


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

    def simulate_relationship(self, sample_size=100000, sparsity_factor=0.001, gen_exp=1.4, init_keep_rate=0.8,
                              **kwargs):
        """
        Create a simulated example of a pedigree and compute its relationship.
        :param sample_size: Size of the cohort.
        :param sparsity_factor: Number of nonzero entries in the IBD matrix.
        :param gen_exp: Gen size = gen_exp X prev gen size.
        :param init_keep_rate: 1 - number of edges to remove before iteration begins.
        :return: A class Pedigree object, and an entry list of the simulation.
        """
        assert sample_size > 0, "Sample size should be a positive number"
        assert (sparsity_factor > 0) and (sparsity_factor < 1), \
            "Sparsity factor must be within the range (0, 1)"
        assert gen_exp > 0, "gen_exp should be a positive number"
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


if __name__ == "__main__":
    args = scilmm_parse_arguments()
    ibd_creator = IBDCompute(**args.__dict__)
