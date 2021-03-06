#!/usr/bin/env python3
"""
t-SNE (stochastic Neightbor Embedding)
"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
      to be transformed by t-SNE
      - n is the number of data points
      - d is the number of dimensions in each point
    * perplexity is the perplexity that all Gaussian distributions
      should have
    * tol is the maximum tolerance allowed (inclusive) for the difference
      in Shannon entropy from perplexity for all Gaussian distributions
    Returns: P, a numpy.ndarray of shape (n, n) containing the symmetric
    P affinities
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        copy = D[i].copy()
        copy = np.delete(copy, i, axis=0)
        Hi, Pi = HP(copy, betas[i])
        Hdiff = Hi - H
        betamin = None
        betamax = None
        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                betamin = betas[i, 0]
                if betamax is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamax) / 2

            else:
                betamax = betas[i, 0]
                if betamin is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (betas[i, 0] + betamin) / 2

            Hi, Pi = HP(copy, betas[i])
            Hdiff = Hi - H
        Pi = np.insert(Pi, i, 0)
        P[i] = Pi

    # Symmetric (P.T + P) / (2n)
    sym_P = (P.T + P) / (2 * n)
    return sym_P
