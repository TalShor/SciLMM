import numpy as np

__author__ = 'tal.shor'


class TopoSort:
    def __init__(self, relationship_mat):
        self.__children_mat = relationship_mat.transpose(True).tocsr()
        self.__marks = np.zeros((relationship_mat.shape[0], 2), dtype=np.bool)
        self.__new_indices = []

    def __visit(self, my_node):
        if self.__marks[my_node, 0] == 1:
            raise Exception("Not a DAG")

        if self.__marks[my_node][0] == 0 and self.__marks[my_node][1] == 0:
            self.__marks[my_node, 0] = 1
            for child in self.__children_mat.getrow(my_node).nonzero()[1]:
                self.__visit(child)
            self.__marks[my_node, 0] = 0
            self.__marks[my_node, 1] = 1
            self.__new_indices.insert(0, my_node)

    def topo_sort(self):
        n_prints = 1
        next_node = 0

        while True:

            left = False

            for i in xrange(next_node, self.__marks.shape[0]):
                if self.__marks[i, 1] == 0:
                    next_node = i
                    left = True
                    break

            if not left:
                return self.__new_indices

            if next_node > n_prints * 100000:
                n_prints += 1
                #print "Topo sort has done : ", next_node

            self.__visit(next_node)
