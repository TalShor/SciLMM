from unittest import TestCase

import numpy as np
import pandas as pd


class Pedigree:
    def __init__(self, pedigree=None, delimiter=' ', null_value='0'):
        """
        Creates a Pedigree class instance.
        :param pedigree: A pedigree instance. If needs to be loaded later keep value as None.
            If used 'FID' column is not necessary.
        :param delimiter: The delimiter of the pedigree file. Default value is ' '.
        :param null_value: Value of null instances in pedigree. Default value is '0'.
        """
        self.delimiter = delimiter
        self.null_value = null_value
        self.family = pedigree
        self.entries = None
        self.all_connections = None

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
        if getattr(self, 'child_' + parent) is None:
            self.get_entries()
            self.__dict__['child_' + parent] = np.array(
                [[self.entries[child], self.entries[parent]] for child, parent in
                 self.family[['IID', 'F_IID']][
                     self.get_non_null_indices(self.family['F_IID'])].values])
        return getattr(self, 'child_' + parent)

    def get_all_parent_child_edges(self):
        if self.all_connections is None:
            self.all_connections = np.vstack(
                (self.get_parent_child_edges('father'), self.get_parent_child_edges('mother')))
        return self.all_connections

#TODO: FAM.py from line 70 on.