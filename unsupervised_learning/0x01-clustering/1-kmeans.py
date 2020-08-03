#!/usr/bin/env python3
"""
adjustment of centroids
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
      - n is the number of data points
      - d is the number of dimensions for each data point
    * k is a positive integer containing the number of clusters
    * iterations is a positive integer containing the maximum
      number of iterations that should be performed
    * If no change occurs between iterations, your function
      should return
    * Initialize the cluster centroids using a multivariate
      uniform distribution (based on0-initialize.py)
    * If a cluster contains no data points during the update
      step, reinitialize its centroid
    * You should use numpy.random.uniform exactly twice
    * You may use at most 2 loops
    * Returns: C, clss, or None, None on failure
      - C is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
      - clss is a numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point
        belongs to
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(k) is not int or k <= 0:
        return None

    if type(iterations) is not int or iterations <= 0:
        return None

    n, d = X.shape
    # ******** Initialize randomly the centroids *********
    min_X = np.min(X, axis=0).astype(np.float)
    max_X = np.max(X, axis=0).astype(np.float)
    centr = np.random.uniform(low=min_X, high=max_X, size=(k, d))

    # *************** runinng iteration times ********************
    for i in range(iterations):
        copy = centr.copy()
        # ********  getting distances **********
        data = X[:, np.newaxis, :]
        aux_centr = copy[np.newaxis, :, :]
        dist = np.linalg.norm((data - aux_centr), axis=2)
        clase = np.argmin(dist, axis=1)

        # ******** adjust centroids **********
        for i in range(k):
            mask = np.where(clase == i)
            new_data = X[mask]
            # if no data exist reinitialize that centroid
            if len(new_data) == 0:
                copy[i] = np.random.uniform(min_X, max_X, (1, d))
            else:
                copy[i] = np.mean(new_data, axis=0)
        if (centr == copy).all():
            break
        else:
            centr = copy

    return (centr, clase)
