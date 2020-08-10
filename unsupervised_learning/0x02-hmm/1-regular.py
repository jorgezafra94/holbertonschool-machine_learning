#!/usr/bin/env python3
"""
Regular Markov chain
"""

import numpy as np


def regular(P):
    """
    * P is a is a square 2D numpy.ndarray of shape (n, n) representing
      the transition matrix
    * P[i, j] is the probability of transitioning from state i to
      state j
    * n is the number of states in the markov chain

    Returns: a numpy.ndarray of shape (1, n) containing the steady state
    probabilities, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    if P.shape[0] != P.shape[1]:
        return None

    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    _, eig_vec = np.linalg.eig(P.T)

    normalization = (eig_vec/eig_vec.sum()).real

    aux = np.dot(normalization.T, P)

    for elem in aux:
        if (elem >= 0).all() and np.isclose(elem.sum(), 1):
            return elem.reshape(1, -1)
    return None
