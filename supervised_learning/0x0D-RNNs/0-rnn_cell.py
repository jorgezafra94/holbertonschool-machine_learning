#!/usr/bin/env python3
"""
creating single RNN cell
"""

import numpy as np


class RNNCell():
    """
    Structure of just one RNN cell
    """
    def __init__(self, i, h, o):
        """
        * i is the dimensionality of the data
        * h is the dimensionality of the hidden state
        * o is the dimensionality of the outputs
        * Wh and bh are for the concatenated hidden state and input data
        * Wy and by are for the output
        * The weights should be initialized using a random normal distribution
          in the order listed above
        * The weights will be used on the right side for matrix multiplication
        * The biases should be initialized as zeros
        """
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains the data input
          for the cell
        * m is the batch size for the data
        * h_prev is a numpy.ndarray of shape (m, h) containing the previous
          hidden state

        Returns: h_next, y
        * h_next is the next hidden state
        * y is the output of the cell
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        a_next = np.tanh(np.dot(xh, self.Wh) + self.bh)
        y_pred = np.dot(a_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        return (a_next, y_pred)
