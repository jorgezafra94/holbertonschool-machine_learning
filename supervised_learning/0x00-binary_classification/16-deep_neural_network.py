#!/usr/bin/env python3
"""
First Deep Neural Network
"""

import numpy as np


class DeepNeuralNetwork():
    """
    Defines a deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Constructor
        nx is the number of input features
        layers is a list representing the number of nodes in
               each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # initializing based on He et al method
        for i in range(len(layers)):

            if (type(layers[i]) is not int):
                raise TypeError("layers must be a list of positive integers")

            keyW = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.weights[keyW] = w
            else:
                aux = np.sqrt(2 / layers[i - 1])
                w = np.random.randn(layers[i], layers[i - 1]) * aux
                self.weights[keyW] = w

            b = np.zeros((layers[i], 1))
            self.weights[keyb] = b
