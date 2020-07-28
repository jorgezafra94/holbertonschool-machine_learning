#!/usr/bin/env python3
"""
t-SNE (stochastic Neightbor Embedding)
Shannon entropy
"""

import numpy as np


def HP(Di, beta):
    """
    * Di is a numpy.ndarray of shape (n - 1,) containing the
      pariwise distances between a data point and all other
      points except itself
      - n is the number of data points
    * beta is the beta value for the Gaussian distribution
    Returns: (Hi, Pi)
    * Hi: the Shannon entropy of the points
    * Pi: a numpy.ndarray of shape (n - 1,) containing the P
      affinities of the points
    """
    Di_aux = Di.copy()

    # original equation of P(ij)
    numerator = np.exp(-Di_aux * beta)
    denominator = np.sum(np.exp(-Di_aux * beta))
    Pi = numerator / denominator

    # equation of H(i)
    Hi = -1 * np.sum(Pi * np.log2(Pi))

    return (Hi, Pi)
