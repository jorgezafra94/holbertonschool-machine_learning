#!/usr/bine/env python3
"""
slice like a ninja
"""


def np_slice(matrix, axes={}):
    """
    slice is a python function that works as [:3] or this filters
    slice(None, None, None)--- is the same as complete array
    @matrix: matrix to filter or slice
    @axes: dictionario that contains the information about what
                  axis and filter should be done in matrix
    Return: - new matrix: the matrix sliced
    """
    listSlice = []
    for i in range(len(matrix.shape)):
        flag = 0
        for key in axes:
            if(i == key):
                flag = 1
                listSlice.append(slice(*axes[key]))
        if(flag == 0):
            listSlice.append(slice(None, None, None))
    tupleSlice = tuple(listSlice)
    new = matrix[tupleSlice]
    return new
