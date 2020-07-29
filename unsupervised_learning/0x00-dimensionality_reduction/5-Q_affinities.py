#!/usr/bin/env python3
"""
t-SNE (Stochastic Neighbor Embedding)
getting Qij
"""

import numpy as np


def Q_affinities(Y):
    """
    * Y is a numpy.ndarray of shape (n, ndim) containing the
      low dimensional transformation of X
      - n is the number of points
    * ndim is the new dimensional representation of X
    Returns: Q, num
    * Q is a numpy.ndarray of shape (n, n) containing
      the Q affinities
    * num is a numpy.ndarray of shape (n, n) containing
      the numerator of the Q affinities
    """
    n, d = Y.shape
    x_square = np.sum(np.square(Y), axis=1)
    y_square = np.sum(np.square(Y), axis=1)
    xy = np.dot(Y, Y.T)
    # getting matrix of distances
    D = np.add(np.add((-2 * xy), x_square).T, y_square)

    # the ecuation here calculates the numerator as the pij
    # the difference is the denominator is the sum of all
    # the distances that is why they use k and l instead of i and j
    Q = np.zeros((n, n))
    num = np.zeros((n, n))
    for i in range(n):
        Di = D[i].copy()
        Di = np.delete(Di, i, axis=0)
        numerator = (1 + Di) ** (-1)
        numerator = np.insert(numerator, i, 0)
        num[i] = numerator

    den = num.sum()
    Q = num / den
    return Q, num
