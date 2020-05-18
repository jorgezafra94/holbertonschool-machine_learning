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
    aux = list(zip(X, Y))
    shuffle = np.random.permutation(aux)
    shu_x = shuffle[::, 0]
    shu_y = shuffle[::, 1]
    return (shu_x, shu_y)
