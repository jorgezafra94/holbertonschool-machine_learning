#!/usr/bin/env python3
"""
bidirectional RNN forward propagation
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    * bi_cells is an instance of BidirectinalCell that
      will be used for the forward propagation
    * X is the data to be used, given as a numpy.ndarray
      of shape (t, m, i)
    * * t is the maximum number of time steps
    * * m is the batch size
    * * i is the dimensionality of the data
    * h_0 is the initial hidden state in the forward
      direction, given as a numpy.ndarray of shape (m, h)
    * * h is the dimensionality of the hidden state
    * h_t is the initial hidden state in the backward
      direction, given as a numpy.ndarray of shape (m, h)

    Returns: H, Y
    * H is a numpy.ndarray containing all of the concatenated hidden states
    * Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape

    HF = np.zeros((t, m, h))
    HB = np.zeros((t, m, h))

    h_prev_p = h_0
    h_next_b = h_t

    for i in range(t):
        x_tf = X[i]
        x_tb = X[-(i + 1)]
        h_prev_p = bi_cell.forward(h_prev_p, x_tf)
        h_next_b = bi_cell.backward(h_next_b, x_tb)

        HF[i] = h_prev_p
        HB[-(i + 1)] = h_next_b

    H = np.concatenate((HF, HB), axis=-1)
    Y = bi_cell.output(H)

    return (H, Y)
