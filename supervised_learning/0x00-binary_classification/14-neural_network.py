#!/usr/bin/env python3
"""
neural network with one hidden layer performing binary classification:
"""


import numpy as np


class NeuralNetwork():
    """
    Here is my first Holberton NeuralNetwork Class
    here we are going to use One hidden layer
    the main things to keep in mind about a neuron is
    the ecuation y = sum(w.x) + b
    where w are the weights in this case W
          x are the inputs in this case nx
          b are the biases
    A is the activated output of the neuron
    """
    def __init__(self, nx, nodes):
        """
        constructor of class
        nx is the number of input features to the neuron
        nodes is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # input layer parameters
        W1 = np.random.randn(nx, nodes)
        self.__W1 = W1.reshape(nodes, nx)
        b1 = np.zeros(nodes)
        self.__b1 = b1.reshape(nodes, 1)
        self.__A1 = 0
        # hidden layer parameters
        W2 = np.random.randn(nodes, 1)
        self.__W2 = W2.reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        return values of Weights first layer
        """
        return self.__W1

    @property
    def W2(self):
        """
        return values of Weights hidden layer
        """
        return self.__W2

    @property
    def b1(self):
        """
        return values of bias first layer
        """
        return self.__b1

    @property
    def b2(self):
        """
        return values of bias hidden layer
        """
        return self.__b2

    @property
    def A1(self):
        """
        return values of Active hidden layer
        """
        return self.__A1

    @property
    def A2(self):
        """
        return values of Active output
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains
             the input data
        nx is the number of input features to the neuron
        m is the number of examples
        Z = sum(w.x) + b
        A = 1/(1 + exp(-z))
        """
        # first layer
        Z1 = (np.matmul(self.__W1, X)) + self.__b1
        A1 = 1/(1 + np.exp(-Z1))
        self.__A1 = A1
        # hidden layer
        Z2 = (np.matmul(self.__W2, A1) + self.__b2)
        A2 = 1/(1 + np.exp(-Z2))
        self.__A2 = A2
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains
             the correct labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
            output of the neuron for each example
        """
        first = np.log(A)
        second = np.log(1.0000001 - A)
        third = 1 - Y
        Lost_Fun = -((Y * first) + (third * second))
        Cost_Fun = (Lost_Fun.sum())/Y.shape[1]
        return Cost_Fun

    def evaluate(self, X, Y):
        """
        Evaluates the neural networks predictions
        """
        _, Active = self.forward_prop(X)
        Final = np.where(Active >= 0.5, 1, 0)
        return (Final, self.cost(Y, Active))

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        d(1/(1+e^-z))/dz = e^-z / (1 + e^-z)
        e^-z / (1 + e^-z) = 1/(1+e^-z) * (1 - 1/(1+e^-z))
        e^-z / (1 + e^-z) = A * (1 - A)
        """
        # recalculating Weights hidden layer
        dW2 = np.matmul(A1, (A2 - Y).transpose())
        dW2 = dW2 / X.shape[1]

        # recalculation bias of output
        db2 = ((A2 - Y).sum(axis=1, keepdims=True)) / X.shape[1]

        # recalculation Weights first layer
        aux1 = np.matmul(self.__W2.transpose(), (A2 - Y))
        aux2 = (A1 * (1 - A1))
        dZ1 = aux1 * aux2
        dW1 = (np.matmul(dZ1, X.transpose())) / X.shape[1]

        # recalculation bias hidden layer
        db1 = dZ1.sum(axis=1,  keepdims=True) / X.shape[1]

        self.__W2 = self.__W2 - (alpha * dW2.transpose())
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron
        epochs = iterations
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return self.evaluate(X, Y)
