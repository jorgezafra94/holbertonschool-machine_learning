#!/usr/bin/env python3
"""
pdf of gaussian distribution
"""

import numpy as np


def pdf(X, m, S):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data
      points whose PDF should be evaluated
    * m is a numpy.ndarray of shape (d,) containing the mean of
      the distribution
    * S is a numpy.ndarray of shape (d, d) containing the
      covariance of the distribution
    * You are not allowed to use any loops
    Returns: P, or None on failure
    * P is a numpy.ndarray of shape (n,) containing the PDF
      values for each data point
    * All values in P should have a minimum value of 1e-300
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    if X.shape[1] != S.shape[1] or S.shape[0] != S.shape[1]:
        return None

    if X.shape[1] != m.shape[0]:
        return None

    _, d = X.shape

    Q = np.linalg.inv(S)
    det = np.linalg.det(S)

    den = np.sqrt(((2 * np.pi) ** d) * det)

    diff = X.T - m[:, np.newaxis]

    M1 = np.matmul(Q, diff)
    M2 = np.sum(diff * M1, axis=0)
    M3 = - M2 / 2

    density = np.exp(M3) / den

    density = np.where(density < 1e-300, 1e-300, density)

    return density
