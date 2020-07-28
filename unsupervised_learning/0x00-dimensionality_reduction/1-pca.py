#!/usr/bin/env python3
"""
Getting Transform
Always is better to use SVD instead of eigendecomposition
"""

import numpy as np


def pca(X, ndim):
    """
    * X is a numpy.ndarray of shape (n, d) where:
      - n is the number of data points
      - d is the number of dimensions in each point
    * ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X
    """
    n, d = X.shape
    # get the covariance matrix always be careful with the axis
    # it depends of n position
    X_mean = X - np.mean(X, axis=0, keepdims=True)

    # getting the SVD
    U, S, VT = np.linalg.svd(X_mean)

    # getting W and Wr
    W = VT.T
    Wr = W[:, :ndim]

    # getting Tr
    Tr = np.dot(X_mean, Wr)
    return Tr
