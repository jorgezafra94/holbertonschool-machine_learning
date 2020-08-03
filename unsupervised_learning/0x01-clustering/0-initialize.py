#!/usr/bin/env python3
"""
Initialization method
"""

import numpy as np


def initialize(X, k):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
      that will be used for K-means clustering
      - n is the number of data points
      - d is the number of dimensions for each data point
    * k is a positive integer containing the number of clusters
    * The cluster centroids should be initialized with a multivariate
      uniform distribution along each dimension in d:
    * The minimum values for the distribution should be the minimum
      values of X along each dimension in d
    * The maximum values for the distribution should be the maximum
      values of X along each dimension in d
    * You should use numpy.random.uniform exactly once
    * You are not allowed to use any loops
    * Returns: a numpy.ndarray of shape (k, d) containing the initialized
      centroids for each cluster, or None on failure
    """
    _, d = X.shape

    # Initialize randomly the centroids
    # using the min values and max values of X
    # getting min values per column
    min_Xvals = np.min(X, axis=0).astype(np.float)

    # getting max values per column
    max_Xvals = np.max(X, axis=0).astype(np.float)

    # creating multivariate uniform distribution using float arrays
    centroids = np.random.uniform(low=min_Xvals, high=max_Xvals, size=(k, d))

    return centroids
