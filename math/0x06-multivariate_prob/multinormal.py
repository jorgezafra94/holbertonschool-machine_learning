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
        * mean - a numpy.ndarray of shape (d, 1) containing the mean
          of data
        * cov - a numpy.ndarray of shape (d, d) containing the covariance
          matrix data
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

    def pdf(self, x):
        """
        * x is a numpy.ndarray of shape (d, 1) containing the data point
          whose PDF should be calculated
          - d is the number of dimensions of the Multinomial instance
        * If x is not a numpy.ndarray, raise a TypeError with the message
          x must be a numpy.ndarray
        * If x is not of shape (d, 1), raise a ValueError with the message
          x must have the shape ({d}, 1)
        * Returns the value of the PDF
        """
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d, _ = self.cov.shape

        if len(x.shape) != 2 or x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        det = np.linalg.det(self.cov)
        first = 1 / (((2 * np.pi) ** (d / 2)) * (np.sqrt(det)))
        second = np.dot((x - self.mean).T, np.linalg.inv(self.cov))
        third = np.dot(second, (x - self.mean))
        pdf = first * np.exp((-1 / 2) * third)
        pdf = pdf[0][0]
        return pdf
