#!/usr/bin/env python3
"""
Using Dropout in the Gradient Descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    * Y is a one-hot numpy.ndarray of shape (classes, m) that contains
         the correct labels for the data
    * classes is the number of classes
    * m is the number of data points
    * weights is a dictionary of the weights and biases of the neural network
    * cache is a dictionary of the outputs and dropout masks of each
          layer of the neural network
    * alpha is the learning rate
    * keep_prob is the probability that a node will be kept
    * L is the number of layers of the network
    * All layers use the tanh activation function except the last
         which uses the softmax activation function
    * The weights of the network should be updated in place
    """
    aux = weights.copy()
    dZ = 0
    m = Y.shape[1]
    for i in range(L, 0, -1):
        key_w = "W{}".format(i)
        key_b = "b{}".format(i)
        key_in = "A{}".format(i)
        X = "A{}".format(i - 1)

        if i == L:
            dZ = (cache[key_in] - Y)
            dW = np.matmul(dZ, cache[X].transpose()) / m
            db = (dZ.sum(axis=1, keepdims=True)) / m

        else:
            deriv = 1 - ((cache[key_in]) * (cache[key_in]))
            weight = aux["W{}".format(i + 1)]

            dropout = cache["D{}".format(i)]
            dZL = np.matmul(weight.transpose(), dZ)
            mask = dZL * dropout
            mask = mask / keep_prob
            dZL = mask * deriv

            dW = (np.matmul(dZL, cache[X].transpose())) / m
            db = (dZL.sum(axis=1, keepdims=True)) / m
            dZ = dZL

        weights[key_w] = aux[key_w] - (alpha * (dW))
        weights[key_b] = aux[key_b] - (alpha * db)
