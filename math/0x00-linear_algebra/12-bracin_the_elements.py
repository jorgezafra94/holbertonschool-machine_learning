#!/usr/bin/env python3
"""
using wise-element operations
"""


def np_elementwise(mat1, mat2):
    """
    using Numpy wise-element operators
    Methods of Numpy arrays or matrices
    suma = np.add(mat1, mat2)
    resta = np.subtract(mat1, mat2)
    multi = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    """
    suma = mat1 + mat2
    resta = mat1 - mat2
    multi = mat1 * mat2
    div = mat1 / mat2
    return (suma, resta, multi, div)
