from scipy.sparse import csr_matrix
import numpy as np
from SparseFunctions import save_sparse_csr

def example1():
    #       0
    #     / | \
    #    /  1  \
    #   2 _/ \_ 3
    # Corr(2,3) should be 3/4 ( 2 paths of size 2 and 2 of size 3)
    capm = csr_matrix((4, 4))
    for i in range(4):
        for j in range(4):
            capm[i, j] = 1

    ancestors = csr_matrix((4, 4))
    ancestors[1, 0] = 1
    ancestors[2, 1] = ancestors[2, 0] = 1
    ancestors[3, 1] = ancestors[3, 0] = 1

    rel = csr_matrix((4, 4))
    rel[1, 0] = 1
    rel[2, 1] = rel[2, 0] = 1
    rel[3, 1] = rel[3, 0] = 1

    res_required = [[[1,0,1,1], [2,0,1,1], [2,0,2,1], [3,0,1,1], [3, 0, 2, 1]],
                    [[2,1,1,1], [2,0,2,1], [3,1,1,1], [3,0,2,1]],
                    [[3,1,2,1], [3,0,2,1], [3,0,3,2]],
                    []]

    return rel, ancestors, capm, res_required


def example2():
    #       0
    #     / |
    #    /  1
    #   2 _/ \_ 3
    # Corr(2,3) should be 3/8 ( 1 path of size 2 and one of 3)

    capm = csr_matrix((4, 4))
    for i in range(4):
        for j in range(4):
            capm[i, j] = 1

    ancestors = csr_matrix((4, 4))
    ancestors[1, 0] = 1
    ancestors[2, 1] = ancestors[2, 0] = 1
    ancestors[3, 1] = ancestors[3, 0] = 1

    rel = csr_matrix((4, 4))
    rel[1, 0] = 1
    rel[2, 1] = rel[2, 0] = 1
    rel[3, 1] = 1

    res_required = [[[1,0,1,1], [2,0,1,1], [2,0,2,1], [3,0,2,1]],
                    [[2,1,1,1], [2,0,2,1], [3,1,1,1]],
                    [[3,1,2,1], [3,0,3,1]],
                    []]

    return rel, ancestors, capm, res_required


def example3():
    #    / 2 \
    #  0       3 - 4 - 5
    #    \ 1 /
    # 5 and 3 should have only a single path, (as well as 4)

    capm = csr_matrix((6, 6))
    for i in range(6):
        for j in range(6):
            capm[i, j] = 1

    ancestors = csr_matrix((6, 6))
    ancestors[1, 0] = 1
    ancestors[2, 0] = 1
    ancestors[3, 0] = ancestors[3, 1] = ancestors[3, 2] = 1
    ancestors[4, 0] = ancestors[4, 1] = ancestors[4, 2] = ancestors[4, 3] = 1
    ancestors[5, 0] = ancestors[5, 1] = ancestors[5, 2] = ancestors[5, 3] = ancestors[5, 4] = 1

    rel = csr_matrix((6, 6))
    rel[1, 0] = 1
    rel[2, 0] = 1
    rel[3, 1] = rel[3, 2] = 1
    rel[4, 3] = 1
    rel[5, 4] = 1

    res_required = [[[1,0,1,1], [2,0,1,1], [3,0,2,2], [4,0,3,2], [5,0,4,2]],
                    [[2,0,2,1], [3,1,1,1], [3,0,3,1], [4,1,2,1], [4,0,4,1], [5,1,3,1], [5,0,5,1]],
                    [[3,2,1,1], [3,0,3,1], [4,2,2,1], [4,0,4,1], [5,2,3,1], [5,0,5,1]],
                    [[4,3,1,1], [5,3,2,1], [4,0,5,2], [5,0,6,2]],
                    [[5,4,1,1]],
                    []]

    return rel, ancestors, capm, res_required


def inbreeding_check():
    # "the problem at http://www.genetic-genealogy.co.uk/Toc115570135.html"
    #     2 \
    # 0 - 3 - 6 - 8
    #   x       x
    # 1 - 4 - 7 - 9
    #     5 /

    capm = csr_matrix((10, 10))
    for i in range(10):
        for j in range(10):
            capm[i, j] = 1

    ancestors = csr_matrix((10, 10))
    for i in [3,4,6,7,8,9]:
        ancestors[i, 0] = ancestors[i, 1] = 1

    for i in [6,8,9]:
        ancestors[i, 2] = ancestors[i, 3] = 1

    for i in [7,8,9]:
        ancestors[i, 5] = ancestors[i, 4] = 1

    for i in [8,9]:
        ancestors[i, 6] = ancestors[i, 7] = 1


    rel = csr_matrix((10, 10))
    rel[3, 0] = rel[4, 0] = 1
    rel[3, 1] = rel[4, 1] = 1
    rel[6, 3] = 1
    rel[7, 4] = 1
    rel[6, 2] = rel[7, 5] = 1
    rel[8, 6] = rel[9, 6] = 1
    rel[8, 7] = rel[9, 7] = 1

    # for the correlation of 9 and 8 only (the interesting part)
    res_required = np.array([[9,0,6,2], [9,1,6,2], [9,6,2,1], [9,7,2,1]])

    return rel, ancestors, capm, res_required


def example5():
    # meant to test inbreedung coeeficients
    # 0 -- 1 -- 2 -- 3
    #   \-----------/

    rel = csr_matrix((4, 4))
    rel[3, 0] = rel[3, 2] = 1
    rel[2, 1] = rel[1, 0] = 1

    return rel, None, None, None


def henderson_example():
    rel = csr_matrix((7, 7))
    rel[2, 0] = 1
    rel[3, 0] = rel[3, 1] = 1
    rel[4, 2] = rel[4, 3] = 1
    rel[5, 0] = rel[5, 3] = 1
    rel[6, 4] = rel[6, 5] = 1

    real = csr_matrix((7,7))
    real[0,0] = 1; real[0,2] = real[0,3] = real[0,4] = 0.5; real[0,5] = 0.75; real[0,6] = 0.625
    real[1,1] = 1; real[1,3] = 0.5; real[1,4] = real[1,5] = real[1.6] = 0.25
    real[2,2] = 1; real[2,3] = 0.25; real[2,4] = 0.625; real[2,5] = 0.375; real[2,6] = 0.5
    real[3,3] = 1; real[3,4] = 0.625; real[3,5] = 0.75; real[3,6] = 0.6875
    real[4,4] = 1.125; real[4,5] = 0.5625; real[4,6] = 0.84375
    real[5,5] = 1.25; real[5,6] = 0.90625
    real[6,6] = 1.28125

    for i in range(7):
        for j in range(i+1, 7):
            real[j, i] = real[i, j]
    L = csr_matrix((7,7))
    L[[0,1,2,3,4,5,6], [0,1,2,3,4,5,6]] = 1
    L[[2, 3, 3, 4, 4, 4, 5, 6, 6, 6], [0, 0, 1, 0, 2, 3, 3, 3, 4, 5]] = 0.5
    L[[4, 5, 6, 6], [1, 1, 1, 2]] = 0.25
    L[[6], [0]] = 0.625
    L[[5], [0]] = 0.75


    return rel, real, L, np.array([1, 1, 0.866025, 0.707107, 0.707107, 0.707107, 0.637378])

	
def cam_example():
	#        0   1
	#       /   / \
	#      2   3   4
	#       \ / \ /
	#        5   6
	#       /     \
	#      7       8
	#               \
	#                9
	
	rel = csr_matrix((10, 10))
	rel[2,0] = rel[3,1] = rel[4,1] = 10
	rel[5,2] = rel[5,3] = rel[6,3] = rel[6,4] = 1
	rel[7,5] = rel[8,6] = rel[9,8] = 1
	
	return rel.astype(np.bool), np.array([5,6])


def lecture_example():
	#          0 - 1         12 - 13         
	#          / | \          /   \
	#     2 - 3  4  5 ----- 14   15 - 16 22 - 23
	#      / \         / \         /  \     |
	#     6   7 - 8   17  18 ---- 19  20 -  24
	#         / | \           |          |
	#        9 10 11          21         25

	rel = csr_matrix((26, 26))
	parents = [[3,0], [3,1], [4,0], [4,1], 
		[5,0], [5,1], [6,2], [6,3], 
		[7,2], [7,3], [9,7], [9,8],
		[10,7], [10,8], [11,7], [11,8],
		[14, 12], [14, 13], [15,12], [15,13],
		[17, 5], [17, 14], [18, 5], [18,14],
		[19, 15], [19, 16], [20, 15], [20, 16],
		[21, 18], [21, 19], [24, 22], [24, 23],
		[25, 20], [25, 24]]
	
	for pc in parents:
		rel[pc[0], pc[1]] = 1
	return rel.astype(np.bool)


if __name__ == "__main__":
	rel = lecture_example()
	save_sparse_csr("DataSets/funckeduptree.npz", rel)
