#!/usr/bin/env python3
"""
 forward propagation for a simple RNN
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    * rnn_cell is an instance of RNNCell that will be used for
      the forward propagation
    * X is the data to be used, given as a numpy.ndarray of
      shape (t, m, i)
    * * t is the maximum number of time steps
    * * m is the batch size
    * * i is the dimensionality of the data
    * h_0 is the initial hidden state, given as a numpy.ndarray
      of shape (m, h)
    * h is the dimensionality of the hidden state

    Returns: H, Y
    * H is a numpy.ndarray containing all of the hidden states
    * Y is a numpy.ndarray containing all of the outputs
    """
    t, _, i = X.shape
    m, h = h_0.shape

    # we add 1 to H to save the h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    H[0] = h_0
    h_prev = h_0

    for i in range(t):
        x_t = X[i]
        h_prev, y_pred = rnn_cell.forward(h_prev, x_t)
        H[i + 1] = h_prev
        Y[i] = y_pred
    return (H, Y)
