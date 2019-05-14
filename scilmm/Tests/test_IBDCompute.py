import os
from os.path import join
from unittest import TestCase

import numpy as np

from scilmm import IBDCompute


class TestIBDCompute(TestCase):
    def setUp(self):
        self.ibd_compute = IBDCompute()

    def test_compute_relationships(self):
        pedigree, entries_list = self.ibd_compute.compute_relationships(
            join(os.path.abspath(os.path.dirname(__file__)), 'Examples', 'relationship_example.csv'))
        self.assertEqual(len(entries_list), 10)
        self.assertEqual(pedigree.family.shape[0], 10)

    def test_simulate_relationship(self):
        pedigree, entries_list = self.ibd_compute.simulate_relationship(sample_size=1000, sparsity_factor=0.01)
        self.assertEqual(pedigree.family.shape[0], 1000)

        # Testing a family with no connections
        self.assertRaises(Exception, self.ibd_compute.simulate_relationship, sample_size=100, sparsity_factor=0.0001)

    def test_compute_ibd(self):
        _, _ = self.ibd_compute.simulate_relationship(sample_size=1000, sparsity_factor=0.01)
        ibd, _, _ = self.ibd_compute.compute_ibd()
        self.assertEqual(ibd.max(), 1)
        self.assertEqual((ibd == ibd.max()).sum(), 1000)
        self.assertGreaterEqual((ibd == 0.5).sum(), 10)
        self.assertLessEqual((ibd == 0.5).sum(), 1000)

    def test_circular_rel(self):
        pedigree, entries_list = self.ibd_compute.compute_relationships(
            join(os.path.abspath(os.path.dirname(__file__)), 'Examples', 'circular_rel.fam'))
        self.assertEqual(pedigree.relationship.sum(), 3)
        self.assertLess(pedigree.relationship.sum(), pedigree.family.shape[0])

    def test_half_and_full_brothers(self):
        pedigree, entries_list = self.ibd_compute.compute_relationships(
            join(os.path.abspath(os.path.dirname(__file__)), 'Examples', 'half_and_full_brothers.fam'))
        ibd, _, _ = self.ibd_compute.compute_ibd()
        self.assertEqual(ibd[(np.argwhere(entries_list == '0_9')[0][0], np.argwhere(entries_list == '0_8')[0][0])],
                         0.25)
        self.assertEqual(ibd[(np.argwhere(entries_list == '0_3')[0][0], np.argwhere(entries_list == '0_4')[0][0])], 0.5)

    def test_too_many_parents(self):
        pedigree, entries_list = self.ibd_compute.compute_relationships(
            join(os.path.abspath(os.path.dirname(__file__)), 'Examples', 'too_many_parents.fam'))
        self.assertEqual(set(entries_list), {'0_10', '0_11', '0_12'})
