#!/usr/bin/env python3
"""
Markov Chain
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    * P is a square 2D numpy.ndarray of shape (n, n) representing
      the transition matrix
    * P[i, j] is the probability of transitioning from state i to
      state j
    * n is the number of states in the markov chain
    * s is a numpy.ndarray of shape (1, n) representing the
      probability of starting in each state
    * t is the number of iterations that the markov chain has been
      through

    Returns: a numpy.ndarray of shape (1, n) representing the
    probability of being in a specific state after t iterations,
    or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None

    if P.shape[0] != P.shape[1] or s.shape[0] != 1:
        return None

    if P.shape[0] != s.shape[1]:
        return None

    if type(t) is not int or t <= 0:
        return None

    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    sum_test = np.sum(s)
    if not np.isclose(sum_test, 1):
        return None

    prob = s.copy()
    for i in range(t):
        # the prob change per iteration
        prob = np.matmul(prob, P)

    return prob
