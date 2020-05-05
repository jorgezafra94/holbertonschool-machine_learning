#!/usr/bin/env python3
"""
Our first Neuron class
single neuron performing binary classification
"""


import numpy as np
import matplotlib.pyplot as plt


class Neuron():
    """
    Here is my first Holberton Neuron Class
    the main things to keep in mind about a neuron is
    the ecuation y = sum(w.x) + b
    where w are the weights in this case W
          x are the inputs in this case nx
          b are the biases
    A is the activated output of the neuron
    """
    def __init__(self, nx):
        """
        constructor of class
        nx is the number of input features to the neuron
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        w = np.random.randn(nx)

        self.__W = np.array([w])
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ return private weights"""
        return self.__W

    @property
    def b(self):
        """ return private bias"""
        return self.__b

    @property
    def A(self):
        """ return private activate"""
        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        X is a numpy.ndarray with shape (nx, m) that contains
          the input data
        nx is the number of input features to the neuron
        m is the number of examples
        The neuron should use a sigmoid activation function
        """
        # this matrix is the result of all dot product between
        # W and X
        Dot_product = np.matmul(self.__W, X)
        # now we add the bias in that way we fulfill the ecuation
        # z = sum_all(w.x) + b
        Z = Dot_product + self.__b
        # now we pass Z array and we use the sigmoid activation function
        result = 1 / (1 + np.exp(-Z))
        self.__A = result
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Y is a numpy.ndarray with shape (1, m) that contains the correct
          labels for the input data
        A is a numpy.ndarray with shape (1, m) containing the activated
          output of the neuron for each example
        To avoid division by zero errors, please use
           1.0000001 - A instead of 1 - A

        Lost Function = -(y*log(a) + (1-y)*log(1.0000001-a))
             this function is for each value
        Cost Function = sum(lost Function)/m
        """
        first = np.log(A)
        second = np.log(1.0000001 - A)
        third = 1 - Y
        result1 = Y * first
        result2 = third * second
        Lost_funct = -(result1 + result2)
        Cost_funct = (Lost_funct.sum())/Y.shape[1]
        return Cost_funct

    def evaluate(self, X, Y):
        """
        Evaluates the neurons predictions
        we are going to call the previous methods
        """
        Act_res = self.forward_prop(X)
        Final = np.where(self.__A >= 0.5, 1, 0)
        Error = self.cost(Y, Act_res)
        return (Final, Error)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        X, Y, A we already know what are they
        alpha is the learning rate
        the ecuations used are the parcial derivatives, we get it
            in paper
        """
        aux = A - Y
        dw = np.matmul(X, aux.transpose())
        dw = dw / X.shape[1]
        self.__W = self.W - (alpha * dw.transpose())
        db = aux.sum() / X.shape[1]
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000,
              alpha=0.05, verbose=True, graph=True, step=100):
        """
        Trains the neuron
        epochs = iterations
        verbose is a boolean that defines whether or not to print
            information about the training. If True,
            print Cost after {iteration} iterations: {cost}
            every step iterations
        graph is a boolean that defines whether or not to graph
            information about the training once the training has
            completed.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        i = 0
        epochs, costs = [], []

        while(i < iterations + 1):
            self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, self.__A)
                epochs.append(i)
                costs.append(cost)
                if verbose is True:
                    w = "Cost after {} iterations: {}".format(i, cost)
                    print(w)
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
            i += 1

        if graph is True:
            plt.plot(epochs, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.suptitle("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
