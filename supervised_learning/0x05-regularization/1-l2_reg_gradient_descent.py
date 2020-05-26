#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    * Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
           correct labels for the data
    * classes is the number of classes
    * m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * cache is a dictionary of the outputs of each layer of the neural network
    * alpha is the learning rate
    * lambtha is the L2 regularization parameter
    * L is the number of layers of the network
    * The neural network uses tanh activations on each layer except the last,
           which uses a softmax activation
    * The weights and biases of the network should be updated in place
    """
    aux = weights.copy()
    dZ = 0
    m = Y.shape[1]
    for i in range(L, 0, -1):
        frobenius = 0
        key_w = "W{}".format(i)
        key_b = "b{}".format(i)
        key_in = "A{}".format(i)
        X = "A{}".format(i - 1)

        if i == L:
            dZ = (cache[key_in] - Y)
            dW = np.matmul(dZ, cache[X].transpose()) / m
            dW_L2 = dW + ((lambtha / m) * aux[key_w])
            db = (dZ.sum(axis=1, keepdims=True)) / m

        else:
            deriv = 1 - (cache[key_in]) ** 2
            weight = aux["W{}".format(i + 1)]
            dZL = np.matmul(weight.transpose(), dZ) * deriv
            dW = (np.matmul(dZL, cache[X].transpose())) / m
            dW_L2 = dW + ((lambtha / m) * aux[key_w])
            db = (dZL.sum(axis=1, keepdims=True)) / m
            dZ = dZL

        weights[key_w] = aux[key_w] - (alpha * (dW_L2))
        weights[key_b] = aux[key_b] - (alpha * db)
