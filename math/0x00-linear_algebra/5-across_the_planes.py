#!/usr/bin/env python3
"""
addition of matrices
"""


def add_matrices2D(mat1, mat2):
    """
    this function adds two matrices with max 2D
    @mat1: first matrix
    @mat2: second matrix
    Return: - new array: the result of the addition
            - None: if the matrices cant be added
    """
    result = []
    if ((len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0]))):
        for i in range(len(mat1)):
            result.append([])
            for j in range(len(mat1[i])):
                result[i].append(mat1[i][j] + mat2[i][j])
        return result
    else:
        return None
