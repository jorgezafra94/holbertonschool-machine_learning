#!/usr/bin/env python3
"""
Decode of One Hot Matrix
"""

import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a vector of labels

    one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
    classes is the maximum number of classes
    m is the number of examples
    Returns: a numpy.ndarray with shape (m, ) containing the numeric
    labels for each example, or None on failure
    """

    # has to be np.ndarray type and has to have info inside
    if type(one_hot) is not np.ndarray or len(one_hot) == 0:
        return None

    # has to be shape 2 => (classes, m)
    if len(one_hot.shape) != 2:
        return None

    # only can contain 1 and  0
    b = np.where((one_hot != 0) & (one_hot != 1), True, False)
    if b.any() is True:
        return None

    # only can has one 1 per column
    b = one_hot.T.sum(axis=1)
    b = np.where(b > 1, True, False)
    if b.any() is True:
        return None

    decodeHot = np.argmax(one_hot.T, axis=1)

    return decodeHot
