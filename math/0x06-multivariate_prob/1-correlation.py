#!/usr/bin/env python3
"""
correlation matrix
"""

import numpy as np


def correlation(C):
    """
    * C is a numpy.ndarray of shape (d, d) containing a covariance matrix
       - d is the number of dimensions
    * If C is not a numpy.ndarray, raise a TypeError with the message C
      must be a numpy.ndarray
    * If C does not have shape (d, d), raise a ValueError with the message
      C must be a 2D square matrix
    * Returns a numpy.ndarray of shape (d, d) containing the correlation
      matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError('C must be a 2D numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    d, _ = C.shape
    # getting all variance because they are in the diag of the cov matrix
    variance = np.diag(C).reshape(1, d)
    # getting all std from variance
    stddev = np.sqrt(variance)
    # creating std combinations in order to get the specified denominator
    matrix_std = np.dot(stddev.T, stddev)
    # divide the covariance matrix between the matrix of combinations
    corr = C / matrix_std
    return corr
