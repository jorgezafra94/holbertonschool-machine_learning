#!/usr/bin/env python3
"""
function to get the matrix dimensions
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
