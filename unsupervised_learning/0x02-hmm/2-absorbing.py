#!/usr/bin/env python3
"""
Absorbing markov chain
"""

import numpy as np


def absorbing(P):
    """
    * P is a is a square 2D numpy.ndarray of shape (n, n)
      representing the transition matrix
    * P[i, j] is the probability of transitioning from
      state i to state j
    * n is the number of states in the markov chain

    Returns: True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False

    if P.shape[0] != P.shape[1]:
        return False

    sum_test = np.sum(P, axis=1)
    for elem in sum_test:
        if not np.isclose(elem, 1):
            return None

    n, _ = P.shape

    D = np.diag(P)

    if (D == 1).all():
        return True

    if (D != 1).all():
        return False

    absorb = np.where(D == 1, 1, 0)

    # we start to find connections from the absorbing states
    for i in range(n):
        idx = np.where(absorb == 1)

        for ind in idx[0]:
            elem = P[:, ind]
            mask = np.where(elem > 0)[0]
            absorb[mask] = True
            if absorb.all():
                return True

    return absorb.all()
