#!/usr/bin/env python3
"""
total intra-cluster variance
"""

import numpy as np


def variance(X, C):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * C is a numpy.ndarray of shape (k, d) containing the centroid
      means for each cluster
    * You are not allowed to use any loops
    Returns: var, or None on failure
    * var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None

    if X.shape[1] != C.shape[1]:
        return None
    if C.shape[0] > X.shape[0]:
        return None

    n, _ = X.shape

    data = X[:, np.newaxis, :]
    centr = C[np.newaxis, :, :]
    dist = (np.square(data - centr)).sum(axis=2)
    # in this case we use the value instead of the index
    mini_per_datapoint = np.amin(dist, axis=1)
    var = np.sum(mini_per_datapoint)

    return var
