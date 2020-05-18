#!/usr/bin/env python3
"""
Normalization (Standard)
"""

import numpy as np


def normalization_constants(X):
    """
    X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features
    Returns: the mean and standard deviation
             of each feature, respectively
    """
    mean = X.sum(axis=0) / X.shape[0]
    variance = (X - mean) ** 2
    variance = variance.sum(axis=0) / X.shape[0]
    stdev = np.sqrt(variance)
    return (mean, stdev)
