from unittest import TestCase

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    from scilmm.Matrices.Relationship import organize_rel
except:
    from Matrices.Relationship import organize_rel


class Pedigree:
    def __init__(self, pedigree=None, delimiter=' ', null_value='0', female_value='2', check_num_parents=True,
                 remove_cycles=True, **kwargs):
        """
        Creates a Pedigree class instance.
        :param pedigree: A pedigree instance. If needs to be loaded later keep value as None.
            If used 'FID' column is not necessary.
        :param delimiter: The delimiter of the pedigree file. Default value is ' '.
        :param null_value: Value of null instances in pedigree. Default value is '0'.
        :param female_value: The value of a female individual in the 'sex' column.
        :param check_num_parents: A boolean whether to remove cases of more than 2 parents.
        :param remove_cycles: A boolean whether to remove cases of cycles.
        """
        self.remove_cycles = remove_cycles
        self.check_num_parents = check_num_parents
        self.delimiter = delimiter
        self.null_value = null_value
        self.female_value = female_value
        self.family = pedigree
        self.entries = None
        self.entries_dict = None
        self.all_connections = None
        self.relationship = None
        self.sex = None
        self.interest = None

    def load_pedigree(self, path):
        """
        Loads a .fam pedigree file.
        This file should be saved in a headless CSV format

        :param path: The path of the input pedigree file.
            The pedigree file consists of the following 6 columns (in this order):
            - Family ID: unique identifier for family.
            - IID: Individual ID, along with family ID creates a unique identifier for every individual.
            - Father IID: unique identifier of father. Could be null (according to null value).
            - Mother IID: unique identifier of mother. Could be null (according to null value).
            - Gender: individual's gender. Values are:
                - '1': male
                - '2': female
                - '0' unknown
            - Phenotype: Should appear as 0 if not of interest, otherwise as 1.
                If all values are 0's then all individuals are considered of interest.
        """
        df = pd.read_csv(path, delimiter=self.delimiter, dtype=str,
                         names=['FID', 'IID', 'F_IID', 'M_IID', 'gender', 'phenotype'])

        # Add family ID to individual's IIDs:
        df['F_IID'][self.get_non_null_indices(df['F_IID'])] = \
            df["FID"][self.get_non_null_indices(df['F_IID'])].map(str) + "_" + \
            df['F_IID'][self.get_non_null_indices(df['F_IID'])]
        df['M_IID'][self.get_non_null_indices(df['M_IID'])] = \
            df["FID"][self.get_non_null_indices(df['M_IID'])].map(str) + "_" + \
            df['M_IID'][self.get_non_null_indices(df['M_IID'])]
        df['IID'] = df["FID"].map(str) + "_" + df['IID']

        self.family = df

    def get_entries(self):
        """
        Get an iid to index dictionary.
        :return: A dictionary from IIDs to indices in pedigree
        """
        TestCase.assertIsNotNone(self.family, "A pedigree instance must be loaded use .load_pedigree")
        if self.entries is None:
            if 'FID' in self.family.columns:
                self.entries = {iid: i for i, iid in enumerate(
                    [x for x in np.unique(np.concatenate(self.family[['IID', 'F_IID', 'M_IID']].values)) if '_' in x])}
            else:
                self.entries = {iid: i for i, iid in enumerate(
                    [x for x in np.unique(np.concatenate(self.family[['IID', 'F_IID', 'M_IID']].values)) if x])}
            self.entries_dict = {i: iid for iid, i in self.entries.items()}
        return self.entries

    def get_non_null_indices(self, series):
        if self.null_value:
            return series != self.null_value
        else:
            return ~series.isna()

    def get_parent_child_edges(self, parent='father'):
        """
        Get the parent child edges from the pedigree
        :param parent: Can either be 'father' or 'mother'
        :return: An array of child,parent arrays
        """
        assert (parent in ['father', 'mother']), "Parent must be either 'father' or 'mother'"
        if not hasattr(self, 'child_' + parent):
            self.get_entries()
            self.__dict__['child_' + parent] = np.array(
                [[self.entries[child], self.entries[parent]] for child, parent in
                 self.family[['IID', parent.upper()[0]+'_IID']][
                     self.get_non_null_indices(self.family[parent.upper()[0]+'_IID'])].values])
        return getattr(self, 'child_' + parent)

    def get_all_parent_child_edges(self):
        """
        Returns all parent child edges from the pedigree.
        :return: An array of child, parent arrays.
        """
        if self.all_connections is None:
            self.all_connections = np.vstack(
                (self.get_parent_child_edges('father'), self.get_parent_child_edges('mother')))
        return self.all_connections

    def get_relationship(self):
        """
        Get a relationship csr matrix.
        :return: The pedigree based boolean csr matrix of the relationships.
        """
        if self.relationship is None:
            all_co = self.get_all_parent_child_edges()
            assert len(all_co) > 0, IOError("There are no family connections in the database")
            all_ids = np.array(list(self.get_entries().keys()))
            self.relationship = csr_matrix((np.ones(all_co.shape[0]), (all_co[:, 0], all_co[:, 1])),
                                           shape=(all_ids.size, all_ids.size), dtype=np.bool)
        return self.relationship

    def get_sex(self):
        """
        Get the sex values individuals.
        :return: A boolean array where True means individual is female.
        """
        TestCase.assertIsNotNone(self.family, "A pedigree instance must be loaded use .load_pedigree")
        if self.sex is None:
            self.sex = self.family['gender'] == self.female_value
        return self.sex

    def get_individuals_of_interest(self, phenotype_of_interest='phenotype', phenotype_of_interest_value='1'):
        """
        Returns the indices of individuals of interest.
        :param phenotype_of_interest: Name of the column of phenotype of interest.
            This is useful in cases when there are several phenotypes of interest.
        :param phenotype_of_interest_value: Value of interesting individuals.
            This can be useful if we want to look at different phenotype values.
        :return: Array of individuals of interest.
        """
        TestCase.assertIsNotNone(self.family, "A pedigree instance must be loaded use .load_pedigree")
        if phenotype_of_interest in self.family.columns:
            if phenotype_of_interest_value:
                self.interest = np.array(
                    [self.get_entries()[entry_id] for entry_id in
                     self.family[self.family['phenotype'] == phenotype_of_interest_value]['IID']])
            else:
                self.interest = np.array(
                    [self.get_entries()[entry_id] for entry_id in self.family[~self.family['phenotype'].isna()]['IID']])
        else:
            self.interest = np.array([])
        if self.interest.size == 0:
            self.interest = None
        return self.interest

    def compute_all_values(self):
        self.get_entries()
        self.get_relationship()
        self.get_sex()
        self.get_individuals_of_interest()
        self.relationship, self.interest = organize_rel(self.relationship, self.interest,
                                                        remove_cycles=self.remove_cycles,
                                                        check_num_parents=self.check_num_parents)

        if self.interest is None:
            self.interest = np.ones((self.relationship.shape[0])).astype(np.bool)
