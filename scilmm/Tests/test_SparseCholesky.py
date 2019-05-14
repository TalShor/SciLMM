from unittest import TestCase
import numpy as np
from numpy.random import random

from scilmm import IBDCompute, run_estimates


class TestSparseCholesky(TestCase):
    def test_run_estimates(self):
        ibd_compute = IBDCompute()
        pedigree, entries_list = ibd_compute.simulate_relationship(sample_size=1000, sparsity_factor=0.01)
        phenotype = pedigree.family['phenotype'].astype(int)
        ibd, _, _ = ibd_compute.compute_ibd()
        cov = pedigree.family.apply(lambda x: 1 if random() > 0.3 else 0, axis=1)
        he_est = run_estimates(ibd, phenotype, cov, reml=False)