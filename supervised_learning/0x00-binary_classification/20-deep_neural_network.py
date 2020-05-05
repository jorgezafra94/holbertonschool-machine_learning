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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # initializing based on He et al method
        for i in range(len(layers)):

            if (type(layers[i]) is not int):
                raise TypeError("layers must be a list of positive integers")

            keyW = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
                self.__weights[keyW] = w
            else:
                aux = np.sqrt(2 / layers[i - 1])
                w = np.random.randn(layers[i], layers[i - 1]) * aux
                self.__weights[keyW] = w

            b = np.zeros((layers[i], 1))
            self.__weights[keyb] = b

    @property
    def cache(self):
        """
        return cache attribute info
        """
        return self.__cache

    @property
    def L(self):
        """
        return L attribute info
        """
        return self.__L

    @property
    def weights(self):
        """
        return weights attribute info
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the DNN
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            keyw = "W{}".format(i + 1)
            keyb = "b{}".format(i + 1)
            keyA = "A{}".format(i)

            W = self.__weights[keyw]
            b = self.__weights[keyb]

            A = self.__cache[keyA]
            Z = np.matmul(W, A) + b

            A = 1/(1 + np.exp(-Z))
            keyA = "A{}".format(i + 1)
            self.__cache[keyA] = A

        return (A, self.__cache)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        """
        first = np.log(A)
        second = np.log(1.0000001 - A)
        third = (1 - Y)
        Lost_fun = -((Y * first) + (third * second))
        Cost = (Lost_fun.sum()) / Y.shape[1]
        return Cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural networks predictions
        """
        A, _ = self.forward_prop(X)
        final = [1 if i >= 0.5 else 0 for i in A[0]]
        final = np.array([final])
        return (final, self.cost(Y, A))
