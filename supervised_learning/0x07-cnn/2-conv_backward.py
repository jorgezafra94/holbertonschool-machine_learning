#!/usr/bin/env python3
"""
Backpropagation using convolution
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    * dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
      the partial derivatives with respect to the unactivated output of
      the convolutional layer
          m is the number of examples
          h_new is the height of the output
          w_new is the width of the output
          c_new is the number of channels in the output
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
          h_prev is the height of the previous layer
          w_prev is the width of the previous layer
          c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
      the kernels for the convolution
          kh is the filter height
          kw is the filter width
    * b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the
      biases applied to the convolution
    * padding is a string that is either same or valid, indicating
      the type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
          sh is the stride for the height
          sw is the stride for the width
    Returns: the partial derivatives with respect to the previous
    layer (dA_prev), the kernels (dW), and the biases (db), respectively
    """
