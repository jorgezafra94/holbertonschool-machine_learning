#!/usr/bin/env python3
"""
t-SNE (Stochastic Neighbor Embedding)
complete t-SNE
"""

import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0,
         iterations=1000, lr=500):
    """
    * X is a numpy.ndarray of shape (n, d) containing the
      dataset to be transformed by t-SNE
      - n is the number of data points
      - d is the number of dimensions in each point
    * ndims is the new dimensional representation of X
    * idims is the intermediate dimensional representation
      of X after PCA
    * perplexity is the perplexity
    * iterations is the number of iterations
    * lr is the learning rate
    * Every 100 iterations, not including 0, print Cost at
      iteration {iteration}: {cost}
    * {iteration} is the number of times Y has been
      updated and {cost} is the corresponding cost
    Returns: Y, a numpy.ndarray of shape (n, ndim) containing
    the optimized low dimensional transformation of X
    * For the first 100 iterations, perform early exaggeration
      with an exaggeration of 4
    * a(t) = 0.5 for the first 20 iterations and 0.8 thereafter
    """

    n, d = X.shape
    PCA = pca(X, idims)
    P = P_affinities(X=PCA, perplexity=perplexity)
    Y = np.random.randn(n, ndims)
    iY = np.zeros((n, ndims))

    # P exaggeration
    P = 4 * P

    for i in range(0, iterations):
        dY, Q = grads(Y, P)
        if i < 20:
            alpha = 0.5
        else:
            alpha = 0.8

        # removing exaggeration
        if (i + 1) == 100:
            P = P / 4

        if (i + 1) % 100 == 0:
            C = cost(P, Q)
            a = 'Cost at iteration {}: {}'.format(i + 1, C)
            print(a)

        temp = Y - (lr * dY) + (alpha * (Y - iY))
        iY = Y
        Y = temp
    return Y
