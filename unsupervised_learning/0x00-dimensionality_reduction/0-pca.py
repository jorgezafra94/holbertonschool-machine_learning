#!/usr/bin/env python3
"""
Getting weights of PCA
using Singular Value Decomposition
Always is better to use SVD instead of eigendecomposition
"""

import numpy as np


def pca(X, var=0.95):
    """
    * X is a numpy.ndarray of shape (n, d) where:
       - n is the number of data points
       - d is the number of dimensions in each point
    * all dimensions have a mean of 0 across all data points
    * var is the fraction of the variance that the PCA
      transformation should maintain
    Returns: the weights matrix, W, that maintains var fraction
    of X‘s original variance
    * W is a numpy.ndarray of shape (d, nd) where nd is the new
      dimensionality of the transformed X
    """

    # U is left singular Vectors
    # S is singular Values
    # V is right singular Vectors is the same W loadings
    # Vh is the transpose of V
    U, S, VT = np.linalg.svd(X)

    # getting array of cumsums
    cumsum_array = np.cumsum(S)

    # the last elem in cumsum is the 100% so to get the
    # threshold we have to multiply it by var
    threshold = cumsum_array[-1] * var

    mask = np.where(cumsum_array < threshold)

    # r is going to be index that is the first in fulfill >= threshold
    r = len(cumsum_array[mask])

    # we add 1 to the maxpos to include the first element
    # that is grater than the threshold

    W = VT.T
    Wr = W[:, :r+1]
    return Wr
