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

    det = np.linalg.det(S)

    first = np.matmul((X - m), np.linalg.inv(S))
    # here instead of realize a dot multi we have to do
    # a element wise to get a (n, d) matrix and then sum
    # over axis 1 to get a (n,) vector
    second = np.sum(first * (X - m), axis=1)
    num = np.exp(second / -2)

    den = np.sqrt(det) * ((2 * np.pi) ** (d/2))
    pdf = num / den

    pdf = np.where(pdf < 1e-300, 1e-300, pdf)
    return pdf
