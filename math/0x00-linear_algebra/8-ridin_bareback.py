#!/usr/bin/env python3
"""
multiplication of two matrices
"""


def mat_mul(mat1, mat2):
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
