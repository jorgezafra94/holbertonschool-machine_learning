#!/usr/bin/env python3
"""
structure bidirectional Cell
"""

import numpy as np


class BidirectionalCell():
    """
    Structure bidirectional Cell
    """

    def __init__(self, i, h, o):
        """
        * i is the dimensionality of the data
        * h is the dimensionality of the hidden states
        * o is the dimensionality of the outputs
        * Whf and bhf are for the hidden states in the forward direction
        * Whb and bhb are for the hidden states in the backward direction
        * Wy and by are for the outputs
        * The weights should be initialized using a random normal
          distribution in the order listed above
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zeros
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains
          the data input for the cell
        * m is the batch size for the data
        * h_prev is a numpy.ndarray of shape (m, h) containing
          the previous hidden state
        * Returns: h_next, the next hidden state
        """
        xh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(xh, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains the
          data input for the cell
        * m is the batch size for the data
        * h_next is a numpy.ndarray of shape (m, h) containing the
          next hidden state
        * Returns: h_pev, the previous hidden state
        """
        xh = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.dot(xh, self.Whb) + self.bhb)
        return h_prev
