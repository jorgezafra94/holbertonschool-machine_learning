#!/usr/bin/env python3
"""
transpose of a matrix
"""


def matrix_transpose(matrix):
    """
    return the transpose of a matrix
    """
    new_matrix = []

    try:
        for i in range(len(matrix[0])):
            new_matrix.append([])
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_matrix[j].append(matrix[i][j])
    except (IndexError, TypeError):
        for i in range(len(matrix)):
            new_matrix.append([])
        for i in range(len(matrix)):
            new_matrix[i].append(matrix[i])
    return new_matrix
