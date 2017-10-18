__author__ = 'tal.shor'


import numpy as np
from scipy.sparse import csr_matrix


def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])


# takes a mxm sub matrix, and expands it to a nxn matrix (n>m)
# so that the matrix is full only in the indices
# example : ([[1,2],[3,4]], [0,2], 3) gives [[1,0,2][0,0,0][3,0,4]]
def expand_sub_matrix(sub_matrix, full_matrix_indices, full_size):
    # example given patch [244, 250] and a 2x2 sub matrix
    # the non empty values of the sub matrix
    data_loc = sub_matrix.nonzero()
    data_rows = data_loc[0]
    data_cols = data_loc[1]

    # the indices in the main matrix (if sub(0,1)==1 do (244,250)=1)
    data_rows_after = full_matrix_indices[data_rows]
    data_cols_after = full_matrix_indices[data_cols]

    values = np.array(sub_matrix[data_rows,data_cols]).reshape(-1)

    return csr_matrix((values, (data_rows_after, data_cols_after)), shape=(full_size, full_size))