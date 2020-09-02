#!/usr/bin/env python3
"""
creating single LSTM cell
Long Short Term Memory
"""

import numpy as np


class LSTMCell():
    """
    structure of a LSTM cell
    """

    def __init__(self, i, h, o):
        """
        * i is the dimensionality of the data
        * h is the dimensionality of the hidden state
        * o is the dimensionality of the outputs
        * Wf and bf are for the forget gate
        * Wu and bu are for the update gate
        * Wc and bc are for the intermediate cell state
        * Wo and bo are for the output gate
        * Wy and by are for the outputs
        * The weights should be initialized using a random normal
          distribution in the order listed above
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zeros
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains the
          data input for the cell
        * m is the batche size for the data
        * h_prev is a numpy.ndarray of shape (m, h) containing the
          previous hidden state
        * c_prev is a numpy.ndarray of shape (m, h) containing the
          previous cell state
        * The output of the cell should use a softmax activation
          function

        Returns: h_next, c_next, y
        * h_next is the next hidden state
        * c_next is the next cell state
        * y is the output of the cell
        """
        xh = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        ft = np.dot(xh, self.Wf) + self.bf
        ft = 1 / (1 + np.exp(-ft))

        # update gate
        ut = np.dot(xh, self.Wu) + self.bu
        ut = 1 / (1 + np.exp(-ut))

        # candidate value
        c_hat = np.tanh(np.dot(xh, self.Wc) + self.bc)

        # cell state
        c_next = ft * c_prev + ut * c_hat

        # output gate
        ot = np.dot(xh, self.Wo) + self.bo
        ot = 1 / (1 + np.exp(-ot))

        # h_next
        h_next = ot * np.tanh(c_next)

        # output
        y_pred = np.dot(h_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

        return (h_next, c_next, y_pred)
