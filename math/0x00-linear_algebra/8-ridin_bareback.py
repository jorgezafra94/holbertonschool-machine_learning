#!/usr/bin/env python3
"""
multiplication of two matrices
"""


def mat_mul(mat1, mat2):
    """
    normal multiplication of two matrices
    @mat1: first matrix
    @mat2: second matrix
    Return: - new matrix: result of the multiplication
            - None: if the matrices dont fulfill the
                 condition to realize the multiplication
    """
    new = []
    if (len(mat1[0]) == len(mat2)):
        for i in range(len(mat1)):
            elements = []
            for k in range(len(mat2[0])):
                multi = 0
                for j in range(len(mat1[i])):
                    multi = multi + (mat1[i][j] * mat2[j][k])
                elements.append(multi)
            new.append(elements)
        return new
    else:
        return None
