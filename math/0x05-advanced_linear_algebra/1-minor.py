#!usr/bin/env python3
"""
creating our method to get the determinant
"""


def determinant(matrix):
    """
    * matrix is a list of lists whose determinant should be calculated
    * Returns the determinant of matrix
    """
    # determinant matrix 0x0 or 1x1
    if type(matrix) is list and len(matrix) == 1:
        if type(matrix[0]) is list and len(matrix[0]) == 0:
            return 1
        if type(matrix[0]) is list and len(matrix[0]) == 1:
            return matrix[0][0]

    # Matrix list of list
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for elem in matrix:
        if type(elem) is not list:
            raise TypeError('matrix must be a list of lists')

    # if Matrix is not square
    size = len(matrix)
    for elem in matrix:
        if len(elem) != size:
            raise ValueError('matrix must be a square matrix')

    if size == 2:
        det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
        return det
    else:
        det = 0
        count = 0
        cof = 1
        while (count < size):
            multy = matrix[0][count]
            multy = multy * cof
            copy = []
            for elem in matrix:
                copy.append(list(elem))
            copy.pop(0)
            new_mat = []
            for elem in copy:
                elem.pop(count)
                new_mat.append(elem)
            mini_det = determinant(new_mat)
            det = det + (multy * mini_det)
            cof = cof * -1
            count += 1
        return det


def minor(matrix):
    """
    * matrix is a list of lists whose minor matrix should be calculated
    * Returns: the minor matrix of matrix
    here the matrix 0x0 cant be realized
    """
    # list of lists
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for elem in matrix:
        if type(elem) is not list:
            raise TypeError('matrix must be a list of lists')

    # if Matrix is not square and matrix 0x0
    size = len(matrix)
    for elem in matrix:
        if len(elem) != size or len(elem) == 0:
            raise ValueError('matrix must be a non-empty square matrix')

    if size == 1:
        return [[1]]

    else:
        minor = []
        for i in range(size):
            row = []
            count = 0
            while (count < size):
                copy = []
                for elem in matrix:
                    copy.append(list(elem))
                copy.pop(i)
                new_mat = []
                for elem in copy:
                    elem.pop(count)
                    new_mat.append(elem)
                row.append(determinant(new_mat))
                count += 1
            minor.append(row)
        return minor
