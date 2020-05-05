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
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # initializing based on He et al method
        for i in range(len(layers)):

            if (type(layers[i]) is not int or layers[i] < 1):
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
        Final = np.where(A >= 0.5, 1, 0)
        return (Final, self.cost(Y, A))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the DNN
        """
        dZ = 0
        aux = {}
        m = Y.shape[1]

        for i in range(self.__L, 0, -1):
            AL = "A{}".format(i)
            AL_1 = "A{}".format(i - 1)
            W = "W{}".format(i)
            b = "b{}".format(i)

            if i == self.__L:
                dZ = cache[AL] - Y
                dW = np.matmul(dZ, cache[AL_1].transpose())
                dW = dW / m
                db = (dZ.sum(axis=1, keepdims=True)) / m

            else:
                deriv = cache[AL] * (1 - cache[AL])
                weight = self.__weights["W{}".format(i + 1)]
                dZL = np.matmul(weight.transpose(), dZ) * deriv
                dW = np.matmul(dZL, cache[AL_1].transpose())
                dW = dW / m
                db = (dZL.sum(axis=1, keepdims=True)) / m
                dZ = dZL

            actualW = self.__weights[W]
            actualb = self.__weights[b]
            aux[W] = actualW - (alpha * dW)
            aux[b] = actualb - (alpha * db)

        self.__weights = aux

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        i = 0
        while(i < iterations):
            _, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            i += 1

        return self.evaluate(X, Y)
