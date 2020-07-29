#!/usr/bin/env python3
"""
t-SNE (Stochastic Neighbor Embedding)
cost
"""

import numpy as np


def cost(P, Q):
    """
    * P is a numpy.ndarray of shape (n, n) containing the P affinities
    * Q is a numpy.ndarray of shape (n, n) containing the Q affinities
    Returns: C, the cost of the transformation
    """
    Q_new = np.where(Q == 0, 1e-12, Q)
    P_new = np.where(P == 0, 1e-12, P)
    C = np.sum(P * np.log(P_new / Q_new))
    return C
