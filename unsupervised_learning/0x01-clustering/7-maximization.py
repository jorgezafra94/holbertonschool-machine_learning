#!/usr/bin/env python3
"""
maximization step of GMM
"""

import numpy as np


def maximization(X, g):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * g is a numpy.ndarray of shape (k, n) containing the posterior
      probabilities for each data point in each cluster
    * You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
    * pi is a numpy.ndarray of shape (k,) containing the updated
      priors for each cluster
    * m is a numpy.ndarray of shape (k, d) containing the updated
      centroid means for each cluster
    * S is a numpy.ndarray of shape (k, d, d) containing the updated
      covariance matrices for each cluster
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None)

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return (None, None, None)

    if X.shape[0] != g.shape[1]:
        return (None, None, None)

    n, d = X.shape
    k, _ = g.shape

    N_soft = np.sum(g, axis=1)
    pi = N_soft / n

    mean = np.zeros((k, d))
    cov = np.zeros((k, d, d))
    for clus in range(k):
        rik = g[clus].reshape(1, -1)
        denomin = N_soft[clus]
        # mean
        mean[clus] = np.dot(rik, X) / denomin
        # cov
        # we have to use element wise first to keep (d, n) by broadcasting
        # then we can use the matrix multiplication to get (d, d) dims
        first = rik * (X - mean[clus]).T
        cov[clus] = np.dot(first, (X - mean[clus])) / denomin

    return (pi, mean, cov)
