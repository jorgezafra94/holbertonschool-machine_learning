#!/usr/bin/env python3
"""
Applying Adam Optimizator
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    that converts a label vector into a one-hot matrix:

    The last dimension of the one-hot matrix must be the number of classes
    Returns: the one-hot matrix
    """
    one_hot = K.utils.to_categorical(y=labels, num_classes=classes)
    return one_hot
