import numpy as np
import os
from os.path import exists
from SideAlgos.SparseFunctions import load_sparse_csr, save_sparse_csr
from scipy.sparse import eye, lil_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components as cc
from SideAlgos.TopoSort import TopoSort
import argparse
from abc import abstractmethod
from datetime import datetime
from itertools import product
import shutil
from sksparse.cholmod import cholesky
import json


def time_decorator(method):
    def main(*args, **kwargs):
        start = datetime.now()
        val = method(*args, **kwargs)
        return val, datetime.now() - start
    return main


def searchunsorted(main, sec):
    main_argsort = np.argsort(main)
    main_sort = np.sort(main)
    return main_argsort[np.searchsorted(main_sort, sec)]


class DataMatrix(object):
    def __init__(self, rel, good_ind, folder):
        self._n = rel.shape[0]
        self._rel = rel
        self._good_ind = good_ind

        self._folder = folder
        if not os.path.exists(self._folder):
            os.mkdir(self._folder)

    @property
    def n(self):
        return self._n

    @property
    def rel(self):
        return self._rel

    @property
    def good_ind(self):
        return self._good_ind

    @abstractmethod
    def get_matrix(self):
        raise NotImplementedError

    @property
    def folder(self):
        return self._folder + "/"

# TODO: have to take only a subset. look in the original code
class DominanceMatrix(DataMatrix):
    def __init__(self, topo_rel, ind_good, num_mat_indices, num_mat_values, folder):
        self._indices = num_mat_indices
        self._values = num_mat_values
        super(DominanceMatrix, self).__init__(topo_rel, ind_good, folder)

    # get_parents_correlation(A_index, A_val, n, i1, j2) gets A[i1,j2]
    def _get_parents_correlation(self, A_index, A_val, p1, p2):
        # instead of a 2d array - do a 1d array and you can use searchsorted
        new_indices = p1 * (self.n + 1) + p2
        values_indices = np.searchsorted(A_index, new_indices)
        # 0 them out - to remove them later
        values_indices[values_indices >= A_index.shape[0]] = 0
        # everyone that does not appear in the array - is now -1
        values_indices[new_indices != A_index[values_indices]] = -1
        # after fix
        values = A_val.copy()[values_indices]
        values[values_indices == -1] = 0

        return values


    def _get_i_and_j(self):
        i1 = np.load(self.folder + "dominance_i1.npy")
        i2 = np.load(self.folder + "dominance_i2.npy")
        j1 = np.load(self.folder + "dominance_j1.npy")
        j2 = np.load(self.folder + "dominance_j2.npy")
        return i1, i2, j1, j2

    def _create_i_and_j(self, rel, A_indices):

        parents_list = np.split(rel.indices, rel.indptr[1:-1])
        parents_matrix = np.zeros(((self.n + 1), 2))
        for i, parents in enumerate(parents_list):
            parents_matrix[i, :min(2, len(parents))] = parents[:min(2, len(parents))]

        iparents, jparents = parents_matrix[A_indices[0]], parents_matrix[A_indices[1]]
        i1, i2, j1, j2 = iparents[:, 0], iparents[:, 1], jparents[:, 0], jparents[:, 1]

        np.save(self.folder + "dominance_i1.npy", i1)
        np.save(self.folder + "dominance_i2.npy", i2)
        np.save(self.folder + "dominance_j1.npy", j1)
        np.save(self.folder + "dominance_j2.npy", j2)

        return i1, i2, j1, j2

    def _create_A_vals(self):
        A_indices, A_values = self._indices, self._values
       
        A_indices = A_indices + 1

        A_index = A_indices[0] * (self.n + 1) + A_indices[1]
        A_index_sort = np.argsort(A_index)
        A_index = A_index[A_index_sort].astype(np.int64)
        A_values = A_values[A_index_sort]
        A_indices = A_indices[:, A_index_sort].astype(np.int64)

        np.save(self.folder + "A_index.npy", A_index)
        np.save(self.folder + "A_values.npy", A_values)
        np.save(self.folder + "A_indices.npy", A_indices)

        return A_index, A_values, A_indices

    def _create_anc_files(self, A_index, A_values, i1, i2, j1, j2):
        asd = self._get_parents_correlation(A_index, A_values, i1, j2)
        np.save(self.folder + "asd.npy", asd)

        ads = self._get_parents_correlation(A_index, A_values, i2, j1)
        np.save(self.folder + "ads.npy", ads)

        ass = self._get_parents_correlation(A_index, A_values, i1, j1)
        np.save(self.folder + "ass.npy", ass)

        add = self._get_parents_correlation(A_index, A_values, i2, j2)
        np.save(self.folder + "add.npy", add)

        return asd, ads, ass, add

    def _get_anc_files(self):
        asd = np.load(self.folder + "asd.npy")
        ads = np.load(self.folder + "ads.npy")
        ass = np.load(self.folder + "ass.npy")
        add = np.load(self.folder + "add.npy")
        return asd, ads, ass, add

    def _get_rel_with_root(self):
        new_rel = lil_matrix((self.n + 1, self.n + 1))
        rel_nonzero = self.rel.nonzero()
        new_rel[rel_nonzero[0] + 1, rel_nonzero[1] + 1] = self.rel[rel_nonzero]
        return new_rel.tocsr()

    def _get_good_indices(self, A_indices, sorted_nodes_of_interest):
        nonzero_0_indices = np.in1d(A_indices[0], sorted_nodes_of_interest)
        nonzero_1_indices = np.in1d(A_indices[1], sorted_nodes_of_interest)
        return nonzero_0_indices * nonzero_1_indices

    def _get_indices_in_final_matrix(self, A_indices, sorted_nodes_of_interest, only_good_indices):
        sub_nonzero_0 = A_indices[0, only_good_indices]
        sub_nonzero_1 = A_indices[1, only_good_indices]
        final_nonzero_0 = searchunsorted(sorted_nodes_of_interest, sub_nonzero_0)
        final_nonzero_1 = searchunsorted(sorted_nodes_of_interest, sub_nonzero_1)
        return final_nonzero_0, final_nonzero_1


    @time_decorator
    def get_matrix(self):

        sorted_nodes_of_interest = self.good_ind
        sorted_nodes_of_interest += 1

        # create a matrix with a new entry without any connections to ease computation.
        new_rel = self._get_rel_with_root()
        A_index, A_values, A_indices = self._create_A_vals()

        # get the A_parent1, parent2 values for all the combinations
        if exists(self.folder + "asd.npy"):
            asd, ads, ass, add = self._get_anc_files()
        else:
            if exists(self.folder + "dominance_i1.npy"):
                i1, i2, j1, j2 = self._get_i_and_j()
            else:
                i1, i2, j1, j2 = self._create_i_and_j(new_rel, A_indices)
            asd, ads, ass, add = self._create_anc_files(A_index, A_values, i1, i2, j1, j2)
            del i1, i2, j1, j2

        # the real dominance value for all the non zero entries of A (only they might have a non zero dominance)
        dominance = ((asd * ads) + (ass * add)) / 4
        del asd, ads, ass, add, A_index

        # get the indices of the nonzero entires between individuals of interest
        only_good_indices = self._get_good_indices(A_indices, sorted_nodes_of_interest)
        sub_dominance = dominance[only_good_indices]
        final_rows, final_cols = self._get_indices_in_final_matrix(A_indices, sorted_nodes_of_interest, only_good_indices)

        del A_values, A_indices, only_good_indices

        # create the final dominance matrix
        D = lil_matrix((sorted_nodes_of_interest.shape[0], sorted_nodes_of_interest.shape[0]))
        nonzero_dominance = sub_dominance != 0
        D[final_rows[nonzero_dominance], final_cols[nonzero_dominance]] = sub_dominance[nonzero_dominance]
        D[np.arange(D.shape[0]), np.arange(D.shape[0])] = 1

        return D.tocsr()


class NumeratorMatrix(DataMatrix):
    # send topo_rel
    def __init__(self, rel, good_ind, folder):
        super(NumeratorMatrix, self).__init__(rel, good_ind, folder)
        self._good_ind = good_ind


    def _LD(self):
        ancestors_list = np.split(self.rel.indices, self.rel.indptr[1:-1])
        L = lil_matrix((self.n,self.n))
        D = np.zeros((self.n))
        F = np.zeros((self.n))

        for i in range(self.n):

            L[i, i] = 1
            ANC = [i]
            i_parents = ancestors_list[i]
            D[i] = 1 - 0.25 * (i_parents.shape[0] + F[i_parents].sum())

            while len(ANC) > 0:
                j = max(ANC)
                j_parents = ancestors_list[j]
                ANC += j_parents.tolist()
                if j_parents.shape[0] > 0:
                    L[i, j_parents] += np.ones((j_parents.shape)) * 0.5 * L[i, j]
                F[i] += (L[i, j] ** 2) * D[j]
                ANC = [x for x in ANC if x != j]
            F[i] -= 1

        D_full = csr_matrix((self.n, self.n))
        D_full[np.arange(self.n), np.arange(self.n)] = D
        
        return L.tocsr(), D_full






    def _numerator_matrix_values(self, num_of_batches=30):

        #TODO: delete this
        if os.path.exists(self.folder + "L.npz"):
            L, D = load_sparse_csr(self.folder + "L.npz"), load_sparse_csr(self.folder  + "D.npz")
        else:
            L, D = self._LD()
            save_sparse_csr(self.folder + "L.npz", L)
            save_sparse_csr(self.folder + "D.npz", D)



        LT = L.transpose(copy=True)
        L_D = L.dot(D)

        stride = int(np.ceil(self.n / float(num_of_batches)))
        all_indices = None
        all_values = None

        all_indices0_list = []
        all_indices1_list = []
        all_values_list = []

        start = datetime.now()

        for i in range(num_of_batches + 1):
            
            subA = L_D[np.arange(i * stride, min((i+1)*stride, self.n))].dot(LT)

            nonzero = subA.nonzero()
            vals = subA.data
            subA.todok()[nonzero[0], nonzero[1]]

            if nonzero[0].shape[0] > 0:
                all_indices0_list += (nonzero[0] + i * stride).tolist()
                all_indices1_list += nonzero[1].tolist()
                all_values_list += vals.tolist()

        return np.vstack((all_indices0_list, all_indices1_list)), np.array(all_values_list)


    @time_decorator
    def get_matrix(self):
        if os.path.exists(self.folder + "all_indices.npy"):
            indices, values = np.load(self.folder + "all_indices.npy"), np.load(self.folder + "all_values.npy")
        else:
            indices, values = self._numerator_matrix_values()
            np.save(self.folder + "all_indices.npy", indices)
            np.save(self.folder + "all_values.npy", values)


        # get only indices in good_ind, and build a new, good matrix


        m = self._good_ind.shape[0]

        good_entries = np.in1d(indices[0], self._good_ind) & np.in1d(indices[1], self._good_ind)
        sub_indices = indices[:, good_entries].copy()
        sub_values = values[good_entries].copy()

        sub_indices[0] = searchunsorted(self._good_ind, sub_indices[0])
        sub_indices[1] = searchunsorted(self._good_ind, sub_indices[1])

	del good_entries


        num_mat = lil_matrix((m, m))


        num_mat[sub_indices[0], sub_indices[1]] = sub_values
        return num_mat.tocsr(), indices, values


class EpistatisMatrix(DataMatrix):
    def __init__(self, rel, num_mat, folder):
        super(EpistatisMatrix, self).__init__(rel, None, folder)
        self._num_mat = num_mat

    @time_decorator
    def get_matrix(self):
        return self._num_mat.multiply(self._num_mat)


class HouseholdMatrix(DataMatrix):
    def __init__(self, topo_rel, ind_good, folder):
        super(HouseholdMatrix, self).__init__(topo_rel, ind_good, folder)

    @time_decorator
    def get_matrix(self):
        # A semi positive matrix.
        x = self.rel.T.dot(self.rel).astype(np.bool)
        x[np.arange(self.n), np.arange(self.n)] = 1
        num_of_com, components = cc(x)
        res = lil_matrix(self.rel.shape)
        for i in range(num_of_com):
            comp = (components == i).nonzero()[0]
            temp = np.array(list(product(comp, comp)))
            a, b = temp[:, 0], temp[:, 1]
            res[a, b] = 1

        return res.tocsr()[self.good_ind][:, self.good_ind]


class CreateMatrices(object):
    def __init__(self, original_rel, good_ind, epis, dom, hh, main_folder, temp_folder, verbose):
        self._epis, self._dom, self._hh = epis, dom, hh
        self._folder = temp_folder
	self._verbose = verbose
	self._main_folder = main_folder

        if os.path.exists(self.relevant_ind_loc):
            relevant_ind = np.load(self.relevant_ind_loc)
            topo_sort = np.load(self.topo_sort_loc)
            good_ind_topo = np.load(self.good_ind_topo_loc)
            topo_rel = original_rel[relevant_ind][:, relevant_ind][topo_sort][:, topo_sort]
        else:
            if verbose:
                print "Relevant ind + Topo order ... ",
            (relevant_ind, topo_sort, topo_rel, good_ind_topo), self._rel_time = \
	            self._relationship_matrix_fitting(original_rel, good_ind)
            np.save(self.relevant_ind_loc, relevant_ind)
            np.save(self.topo_sort_loc, topo_sort)
            np.save(self.good_ind_topo_loc, good_ind_topo)
            if verbose: 
                print "Successful"


        self._rel_nnz = topo_rel.nnz
        if verbose:
            print "Numerator Matrix ... ",
        (num_mat, topo_elements_indices, topo_elements_values), self._num_time = NumeratorMatrix(topo_rel, good_ind_topo, self._folder).get_matrix()
        self._num_nnz = num_mat.nnz
        save_sparse_csr(self.num_mat_final_loc, num_mat)
        if verbose:
            print "Successful"

        if epis:
            if verbose:
                print "Epistatis Matrix ... ",
            epis_mat, self._epis_time = EpistatisMatrix(topo_rel, num_mat, self._folder).get_matrix()
            self._epis_nnz = epis_mat.nnz
            save_sparse_csr(self.epis_mat_final_loc, epis_mat)
            if verbose:
                print "Successful"

        if hh:
            if verbose:
                print "Household Matrix ... ",
            #hh_mat, self._hh_time = HouseholdMatrix(topo_rel, good_ind_topo, self._folder).get_matrix()
            #self._hh_nnz = hh_mat.nnz
            #save_sparse_csr(self.hh_mat_final_loc, hh_mat)
            self._hh_nnz, self._hh_time = 0, datetime.now() - datetime.now()
            if verbose:
                print "Successful"

        if dom:
            if verbose: 
                print "Dominance matrix ... ",
            dom_mat, self._dom_time = DominanceMatrix(topo_rel, good_ind_topo, topo_elements_indices, topo_elements_values, self._folder).get_matrix()
            self._dom_nnz = dom_mat.nnz
            save_sparse_csr(self.dom_mat_final_loc, dom_mat)
            if verbose:
               print "Successful"
        try:
            cholesky(num_mat)
            #if epis:
            #    cholesky(epis_mat)
            #if dom:
            #    cholesky(dom_mat)
            #if hh:
            #    cholesky(hh_mat)
        except Exception as e:
            print "error", e
      
    @time_decorator
    def _relationship_matrix_fitting(self, original_rel, good_ind):
        relevant_ind = self._important_indices(original_rel, good_ind)
        relevant_rel = original_rel[relevant_ind][:, relevant_ind]

        topo_sort = TopoSort(relevant_rel).topo_sort()
        topo_rel = relevant_rel[topo_sort][:, topo_sort]

        good_ind_topo = searchunsorted(relevant_ind[topo_sort], good_ind)

        return relevant_ind, topo_sort, topo_rel, good_ind_topo

    def _important_indices(self, original_rel, ind_good):
        # am is the matrix containing all the ancestors of the nodes
        am = original_rel.copy()
        rel_copy = original_rel.copy()
        while rel_copy.nnz > 0:
            rel_copy = rel_copy.dot(original_rel)
            am = am + rel_copy

        # ancestors with children (with more than 1 kids)
        awc = np.unique(
            np.concatenate((am[ind_good].nonzero()[1],
                            ind_good)))

        am = am[awc][:, awc]

        i_n = eye(am.shape[0])
        # cam is the matrix where every row represents which nonzero relationships does an individual have? (includes his own)
        cam = (am + i_n).dot((am + i_n).transpose(copy=True))
        ind_good_in_awc = searchunsorted(awc, ind_good)
        # with a non-trivial connection
        relevant_in_cam = (cam.astype(np.bool)[ind_good_in_awc].sum(axis=0) > 1).nonzero()[1]
        # those of interest without parents are still interesting
        ind_good_with_relevants = np.unique(np.concatenate((ind_good_in_awc, relevant_in_cam)))
        return awc[ind_good_with_relevants]

    @property
    def rel_time(self):
        return self._rel_time

    @property
    def num_time(self):
        return self._num_time

    @property
    def epis_time(self):
        return self._epis_time

    @property
    def dom_time(self):
        return self._dom_time

    @property
    def hh_time(self):
        return self._hh_time

    @property
    def rel_nnz(self):
        return self._rel_nnz

    @property
    def num_nnz(self):
        return self._num_nnz

    @property
    def epis_nnz(self):
        return self._epis_nnz

    @property
    def dom_nnz(self):
        return self._dom_nnz

    @property
    def hh_nnz(self):
        return self._hh_nnz

    @property
    def relevant_ind_loc(self):
        return self._folder + "/RelevantInd.npy"

    @property
    def topo_sort_loc(self):
        return self._folder + "/TopoSort.npy"

    @property
    def good_ind_topo_loc(self):
        return self._folder + "/IndInTopoSort.npy"

    @property
    def num_mat_loc(self):
        return self._folder + "/NumeratorMatrix.npz"

    @property
    def epis_mat_loc(self):
        return self._folder + "/EpistatisMatrix.npz"

    @property
    def hh_mat_loc(self):
        return self._folder + "/HouseholdMatrix.npz"

    @property
    def dom_mat_loc(self):
        return self._folder + "/DominanceMatrix.npz"


    @property
    def num_mat_final_loc(self):
        return self._main_folder + "/NumeratorMatrix.npz"

    @property
    def epis_mat_final_loc(self):
        return self._main_folder + "/EpistatisMatrix.npz"

    @property
    def hh_mat_final_loc(self):
        return self._main_folder + "/HouseholdMatrix.npz"

    @property
    def dom_mat_final_loc(self):
        return self._main_folder + "/DominanceMatrix.npz"



def create_matrices(rel_path, main_folder, temp_folder, indices_array=None, epis=False, dom=False, hh=False, verbose=False):
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)

    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)

    rel =  load_sparse_csr(rel_path)
    if indices_array is None:
        ind = np.arange(rel.shape[0])
    else:
        ind = np.load(indices_array)

    c = CreateMatrices(rel, ind, epis, dom, hh, main_folder, temp_folder, verbose)

    results = {}
    results["n"] = rel.shape[0]
    results["rel_nonzero"] = c.rel_nnz
    results["num_nonzero"] = c.num_nnz
    results["num_time"] = c.num_time.total_seconds()
    results["num_path"] = c.num_mat_final_loc
    if epis:
        results["epis_nonzero"] = c.epis_nnz
        results["epis_time"] = c.epis_time.total_seconds()
        results["epis_path"] = c.epis_mat_final_loc
    if dom:
        results["dom_nonzero"] = c.dom_nnz
        results["dom_time"] = c.dom_time.total_seconds()
        results["dom_path"] = c.dom_mat_final_loc
    if epis:
        results["hh_nonzero"] = c.hh_nnz
        results["hh_time"] = c.hh_time.total_seconds()
        results["hh_path"] = c.hh_mat_final_loc

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute sigams')
    parser.add_argument('--epis', dest='epis', action='store_true', default=False, help='Create an epistatis covariance matrix')
    parser.add_argument('--dom', dest='dom', action='store_true', default=False, help='Create a dominance covariance matrix')
    parser.add_argument('--hh', dest='hh', action='store_true', default=False, help='Create a household effect covariance matrix')
    parser.add_argument('--rel_mat', dest='rel', default=None, help='It runs a small example if this flag is not used')
    parser.add_argument('--indices_array', dest='ind', default=None)
    parser.add_argument('--main_folder', dest='main_folder', required=True, help='This is where the files would be stored')
    parser.add_argument('--temp_folder', dest='temp_folder', required=True, help='Some stages require a lot of space (not everything can be handled only in memory - so this is a location to dump temp files')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False)
    args = parser.parse_args()
	
    # run a simple example
    if args.rel is None:
        from SideAlgos.GraphExamples import inbreeding_check
        rel, _, _, _ = inbreeding_check()
        ind = np.array([7, 8,9])
        #ind = np.arange(10)
        rel_loc, ind_loc = os.path.join(args.main_folder, "sample_tree.npz"), os.path.join(args.main_folder, "sample_tree_indices.npy")
        if not os.path.exists(args.main_folder):
            os.mkdir(args.main_folder)
        if not os.path.exists(args.temp_folder):
            os.mkdir(args.temp_folder)

	save_sparse_csr(rel_loc, rel)
        np.save(ind_loc, ind)
    else:
        rel_loc, ind_loc = args.rel, args.ind

    print json.dumps(create_matrices(rel_loc, args.main_folder, args.temp_folder, 
					ind_loc, args.epis, args.dom, args.hh, args.verbose), indent=4)

