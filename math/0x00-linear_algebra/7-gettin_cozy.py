#!/usr/bin/env python3
"""
Gettin’ Cozy
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function that concatenate two matrices of 2D
    @mat1: first matrix
    @mat2: second matrix
    @axis: dimension in wich we are going to realize the concatenation
    Return: - new matrix: if the concatenation could be
                 maden in the axis specified
            - None: if the concatenation couldnt  be maden
    """
    new = []
    if (len(mat1[0]) == len(mat2[0]) and axis == 0):
        new += [elem.copy() for elem in mat1]
        new += [elem.copy() for elem in mat2]
        return new
    elif(len(mat1) == len(mat2) and axis == 1):
        new = [mat1[i] + mat2[i] for i in range(len(mat1))]
        return new
    else:
        return None
