import os
from unittest import TestCase

import numpy as np
from networkx.exception import NetworkXUnfeasible
from nose.tools import assert_raises, assert_equal

from scilmm import SciLMM, load_sparse_csr


class TestSciLMM(TestCase):
    output_folder = './Examples/Tests'

    def test_circular_family_tree(self):
        assert_raises(NetworkXUnfeasible,
                      SciLMM, fam='./Examples/Tests/circular_rel.fam', output_folder=self.output_folder)

        SciLMM(fam=os.path.join(self.output_folder, 'circular_rel.fam'), output_folder=self.output_folder,
               remove_cycles=True)
        entries_list = np.load(os.path.join(self.output_folder, "entries_ids.npy"))
        assert_equal(set(entries_list), {'0', '0_5', '0_6'})

    def test_half_full_brothers(self):
        SciLMM(fam=os.path.join(self.output_folder, 'half_and_full_brothers.fam'), output_folder=self.output_folder,
               ibd=True)
        entries_list = np.load(os.path.join(self.output_folder, "entries_ids.npy"))
        ibd = load_sparse_csr(os.path.join(self.output_folder, "IBD.npz"))
        assert_equal(ibd[(np.argwhere(entries_list == '0_9')[0][0], np.argwhere(entries_list == '0_8')[0][0])], 0.25)
        assert_equal(ibd[(np.argwhere(entries_list == '0_3')[0][0], np.argwhere(entries_list == '0_4')[0][0])], 0.5)

    def test_more_than_two_parents(self):
        SciLMM(fam=os.path.join(self.output_folder, 'too_many_parents.fam'), output_folder=self.output_folder,
               ibd=True, check_num_parents=True)
        entries_list = np.load(os.path.join(self.output_folder, "entries_ids.npy"))
        assert_equal(set(entries_list), {'0_10', '0_11', '0_12'})
