#!/usr/bin/env python3
"""
addition of matrices
"""


def add_matrices2D(mat1, mat2):
    result = []
    if ((len(mat1) == len(mat2)) and (len(mat1[0]) == len(mat2[0]))):
        for i in range(len(mat1)):
            result.append([])
            for j in range(len(mat1[i])):
                result[i].append(mat1[i][j] + mat2[i][j])
        return result
    else:
        return None
