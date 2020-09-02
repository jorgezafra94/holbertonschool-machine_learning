#!/usr/bin/env python3
"""
creating single GRU cell
GRU (GATED RECURRENT UNIT )
"""

import numpy as np


class GRUCell():
    """
    Structure of just one GRU cell
    """
    def __init__(self, i, h, o):
        """
        * i is the dimensionality of the data
        * h is the dimensionality of the hidden state
        * o is the dimensionality of the outputs
        * Wz and bz are for the update gate
        * Wr and br are for the reset gate
        * Wh and bh are for the intermediate hidden state
        * Wy and by are for the output
        * The weights should be initialized using a random normal
          distribution in the order listed above
        * The weights will be used on the right side for matrix
          multiplication
        * The biases should be initialized as zero
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        * x_t is a numpy.ndarray of shape (m, i) that contains the data
          input for the cell
        * m is the batche size for the data
        * h_prev is a numpy.ndarray of shape (m, h) containing the
          previous hidden state

        Returns: h_next, y
        * h_next is the next hidden state
        * y is the output of the cell
        """
        xh = np.concatenate((h_prev, x_t), axis=1)

        # rt = reset gate
        rt = np.dot(xh, self.Wr) + self.br
        rt = 1 / (1 + np.exp(-rt))

        # zt = update gate
        zt = np.dot(xh, self.Wz) + self.bz
        zt = 1 / (1 + np.exp(-zt))

        # h_hat = current state
        r_h = rt * h_prev
        r_hx = np.concatenate((r_h, x_t), axis=1)
        h_hat = np.tanh(np.dot(r_hx, self.Wh) + self.bh)

        # h_next
        # original
        # h_next = ((1 - zt) * h_hat) + (zt * h_prev)

        # alexa equation
        h_next = zt * h_hat + (1 - zt) * h_prev
        # output GRU cell
        y_pred = np.dot(h_next, self.Wy) + self.by
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)

        return (h_next, y_pred)