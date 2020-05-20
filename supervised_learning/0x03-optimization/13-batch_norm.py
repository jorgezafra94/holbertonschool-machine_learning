#!/usr/bin/env python3
"""
Batch Normalization
"""


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n) containing the
               scales used for batch normalization
    beta is a numpy.ndarray of shape (1, n) containing the
             offsets used for batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """
    m = Z.sum(axis=0) / Z.shape[0]
    variance = (Z - m) ** 2
    variance = variance.sum(axis=0) / Z.shape[0]
    Z_norm = (Z - m) / ((variance + epsilon) ** (1/2))
    Z_final = (gamma * Z_norm) + beta
    return Z_final
