#!/usr/bin/env python3
"""
Expectation step
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * pi is a numpy.ndarray of shape (k,) containing the priors for
      each cluster
    * m is a numpy.ndarray of shape (k, d) containing the centroid
      means for each cluster
    * S is a numpy.ndarray of shape (k, d, d) containing the
      covariance matrices for each cluster
    * You may use at most 1 loop
    Returns: posterior, prob_gmm, or None, None on failure
    * posterior is a numpy.ndarray of shape (k, n) containing
      the posterior probabilities for each data point in each cluster
    * prob_gmm is the total log likelihood
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None)

    if type(m) is not np.ndarray or len(m.shape) != 2:
        return (None, None)

    if type(S) is not np.ndarray or len(S.shape) != 3:
        return (None, None)

    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return (None, None)

    # every value of pi should be 0 <= pi <= 1
    mask1 = np.where(pi < 0, True, False)
    mask2 = np.where(pi > 1, True, False)
    if mask1.any() or mask2.any():
        return (None, None)

    if X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2]:
        return (None, None)

    if X.shape[1] != m.shape[1]:
        return (None, None)

    if m.shape[0] != S.shape[0]:
        return (None, None)

    if pi.shape[0] != m.shape[0]:
        return (None, None)

    n, d = X.shape
    k = pi.shape[0]

    # adequate dimensions
    if m.shape[1] != d or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if S.shape[0] != k:
        return None, None

    # sum of pi equal to 1
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    sum_i = 0
    num = np.zeros((k, n))
    for i in range(k):
        num[i] = pi[i] * pdf(X, m[i], S[i])
        sum_i += num[i]

    sum_i = np.sum(num, axis=0, keepdims=True)

    g = num / sum_i

    log_likelihood = np.sum(np.log(sum_i))
    return g, log_likelihood