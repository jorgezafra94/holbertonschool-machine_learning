#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""

import numpy as np


class MultiNormal():
    """
    represents a Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        * data is a numpy.ndarray of shape (d, n) containing the data set:
           - n is the number of data points
           - d is the number of dimensions in each data point
        * If data is not a 2D numpy.ndarray, raise a TypeError with the
          message data must be a 2D numpy.ndarray
        * If n is less than 2, raise a ValueError with the message data
          must contain multiple data points
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        d, n = data.shape
        # also know as E[x] in this case axis=1
        self.mean = np.mean(data, axis=1, keepdims=True)
        X_mean = data - self.mean
        # 1/(n - 1) * ((X-mean) * (X - mean)T)
        self.cov = np.dot(X_mean, X_mean.T) / (n - 1)
