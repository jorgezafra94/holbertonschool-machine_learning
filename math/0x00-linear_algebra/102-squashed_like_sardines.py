#!/usr/bin/env python3
"""
concatenate two matrices with recursion
"""


def matrix_shape(matrix):
    result = []
    try:
        if(len(matrix)):
            aux = matrix_shape(matrix[0])
            result.append(len(matrix))
            result = result + aux
            return result
        result.append(len(matrix))
        return result
    except (IndexError, TypeError):
        return []


def rec_conc(m1, m2, axis):
    new = []
    if axis == 0:
        new += [elem for elem in m1]
        new += [elem for elem in m2]
        return new
    else:
        for i in range(len(m1)):
            result = rec_conc(m1[i], m2[i], axis-1)
            new.append(result)
        return new


def cat_matrices(mat1, mat2, axis=0):
    first = matrix_shape(mat1)
    second = matrix_shape(mat2)
    if (len(first) < axis or len(second) < axis):
        return None
    del first[axis]
    del second[axis]
    if (first == second):
        return rec_conc(mat1, mat2, axis)
    else:
        return None
