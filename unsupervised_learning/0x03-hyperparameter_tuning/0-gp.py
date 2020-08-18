#!/usr/bin/env python3
"""
Initializing
"""

import numpy as np


class GaussianProcess():
    """
    Gaussian Process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        * X_init is a numpy.ndarray of shape (t, 1) representing the
          inputs already sampled with the black-box function
        * Y_init is a numpy.ndarray of shape (t, 1) representing the
          outputs of the black-box function for each input in X_init
          - t is the number of initial samples
        * l is the length parameter for the kernel
        * sigma_f is the standard deviation given to the output of
          the black-box function
        * Sets the public instance attributes X, Y, l, and sigma_f
          corresponding to the respective constructor inputs
        * Sets the public instance attribute K, representing the
          current covariance kernel matrix for the Gaussian process

        Public instance method def kernel(self, X1, X2): that calculates
        the covariance kernel matrix between two matrices:
        - X1 is a numpy.ndarray of shape (m, 1)
        - X2 is a numpy.ndarray of shape (n, 1)
        - the kernel should use the Radial Basis Function (RBF)

        Returns: the covariance kernel matrix as a numpy.ndarray of
        shape (m, n)
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        '''
        Isotropic squared exponential kernel. Computes
        a covariance matrix from points in X1 and X2.
        * X1 numpy.ndarray of shape (t, 1).
        * X2 numpy.ndarray of shape (t, 1).

        Returns Covariance matrix (m x n).
        '''
        first = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        second = np.sum(X2 ** 2, axis=1)
        third = -2 * np.dot(X1, X2.T)

        sqdist = first + second + third

        kernel_1 = (self.sigma_f ** 2)
        kernel_2 = np.exp(-0.5 / self.l ** 2 * sqdist)
        kernel = kernel_1 * kernel_2
        return kernel
