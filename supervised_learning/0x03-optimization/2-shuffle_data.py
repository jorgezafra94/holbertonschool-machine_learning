#!/usr/bin/env python3
"""
Shuffle of Matrix
"""


import numpy as np


def shuffle_data(X, Y):
    """
    X is the first numpy.ndarray of shape (m, nx) to shuffle
    m is the number of data points
    nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
    m is the same number of data points as in X
    ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    vector = np.random.permutation(np.arange(X.shape[0]))
    X_shu, Y_shu = X, Y
    X_shu[np.arange(X.shape[0])] = X[vector]
    Y_shu[np.arange(Y.shape[0])] = Y[vector]
    return X_shu, Y_shu
