#!/usr/bin/env python3
"""
t-SNE (stochastic Neightbor Embedding)
* initializes all variables required to calculate
  the P affinities in t-SNE
"""

import numpy as np


def P_init(X, perplexity):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
      to be transformed by t-SNE
       - n is the number of data points
       - d is the number of dimensions in each point
    * perplexity is the perplexity that all Gaussian distributions
      should have
    Returns: (D, P, betas, H)
    * D: a numpy.ndarray of shape (n, n) that calculates the pairwise
      distance between two data points
    * P: a numpy.ndarray of shape (n, n) initialized to all 0‘s that
     will contain the P affinities
    * betas: a numpy.ndarray of shape (n, 1) initialized to all 1’s
      that will contain all of the beta values
    * H is the Shannon entropy for perplexity perplexity
    """

    n, d = X.shape
    # euclidian distance (x - y)^2  = x^2 + y^2 -2xy
    x_square = np.sum(np.square(X), axis=1)
    y_square = np.sum(np.square(X), axis=1)
    xy = np.dot(X, X.T)
    # we do the Transpose in order to multiply all y^2 respectively
    D = np.add(np.add((-2 * xy), x_square).T, y_square)

    P = np.zeros((n, n))

    betas = np.ones((n, 1))

    H = np.log2(perplexity)

    return (D, P, betas, H)
