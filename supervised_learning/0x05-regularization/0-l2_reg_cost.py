#!/usr/bin/env python3
"""
L2 regularizacion Cost
"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    * cost is the cost of the network without L2 regularization
    * lambtha is the regularization parameter
    * weights is a dictionary of the weights and biases (numpy.ndarrays)
              of the neural network
    * L is the number of layers in the neural network
    * m is the number of data points used
    * Returns: the cost of the network accounting for L2 regularization
    """
    frobenius = 0
    for i in range(1, L + 1):
        key_w = "W{}".format(i)
        w = np.linalg.norm(weights[key_w])
        frobenius += w

    cost_L2 = cost + (lambtha / (2 * m)) * frobenius
    return cost_L2
