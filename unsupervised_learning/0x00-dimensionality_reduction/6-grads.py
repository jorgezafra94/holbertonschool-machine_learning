#!/usr/bin/env python3
"""
t-SNE (Stochastic Neighbor Embedding)
getting gradient descent
"""

import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    * Y is a numpy.ndarray of shape (n, ndim) containing
      the low dimensional transformation of X
    * P is a numpy.ndarray of shape (n, n) containing
      the P affinities of X
    Returns: (dY, Q)
    * dY is a numpy.ndarray of shape (n, ndim) containing
      the gradients of Y
    * Q is a numpy.ndarray of shape (n, n) containing
      the Q affinities of Y
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)

    # equation is dy = sum((P - Q) * (Yi - Yj) * (num))
    # as they are element wise multy we can change the order
    equation1 = ((P - Q) * num)
    dY = np.zeros((n, ndim))
    # *************************** step by step **************
    # for i in range(n):
    #    aux = np.tile(equation1[:, i].reshape(-1, 1), 2)
    #    dY[i] = (aux * (Y[i] - Y)).sum(axis=0)
    for i in range(n):
        # we have to reshape aux because it is (2500,) shape
        aux = np.tile(equation1[:, i], (ndim, 1)).T
        # aux = np.tile(equation1[:, i].reshape(-1, 1), 2)
        # after this process we get a shape (2500, 2) now we can
        # do the multiplication
        dY[i] = (aux * (Y[i] - Y)).sum(axis=0)
    return (dY, Q)
