#!/usr/bin/env python3
"""
add two matrices if they have the same shape otherwise return None
"""


def matrix_shape(matrix):
    """
    return the shape of matrix
    """
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


def mat_add(m1, m2):
    """
    function that realizes the addition recursively of two matrices
    """
    if type(m1[0]) is not list:
        new = [m1[i] + m2[i]for i in range(len(m1))]
        return new
    else:
        new = []
        for i in range(len(m1)):
            result = mat_add(m1[i], m2[i])
            new.append(result)
        return new


def add_matrices(mat1, mat2):
    """
    function that realize the addition between two matrices
    @mat1: first matrix
    @mat2: second matrix
    Return: - new matrix: the addition of these two
            - None: if the matrix cant be added
    """
    first = matrix_shape(mat1)
    second = matrix_shape(mat2)
    if (first == second):
        return mat_add(mat1, mat2)
    else:
        return None
