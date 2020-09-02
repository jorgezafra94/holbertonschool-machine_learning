#!/usr/bin/env python3
"""
forward propagation of deep RNN
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    * rnn_cells is a list of RNNCell instances of length l that
      will be used for the forward propagation
    * l is the number of layers
    * X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    * * t is the maximum number of time steps
    * * m is the batch size
    * * i is the dimensionality of the data
    * h_0 is the initial hidden state, given as a numpy.ndarray
      of shape (l, m, h)
    * h is the dimensionality of the hidden state

    Returns: H, Y
    * H is a numpy.ndarray containing all of the hidden states
    * Y is a numpy.ndarray containing all of the outputs
    """
    l, m, h = h_0.shape
    t, m, i = X.shape
    o = rnn_cells[-1].by.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    H[0] = h_0
    count = 0

    for i in range(t):
        x_t = X[i]
        h_temp = np.zeros((l, m, h))

        for j in range(l):
            h_prev = H[count][j]
            h_prev, output = rnn_cells[j].forward(h_prev, x_t)
            x_t = h_prev
            h_temp[j] = h_prev

        Y[i] = output
        H[i + 1] = h_temp
        count += 1

    return (H, Y)
