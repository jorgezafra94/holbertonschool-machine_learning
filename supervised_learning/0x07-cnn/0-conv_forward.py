#!/usr/bin/env python3
"""
Forward Propagation with Convolution
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
         m is the number of examples
         h_prev is the height of the previous layer
         w_prev is the width of the previous layer
         c_prev is the number of channels in the previous layer
    * W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
      the kernels for the convolution
         kh is the filter height
         kw is the filter width
         c_prev is the number of channels in the previous layer
         c_new is the number of channels in the output
    * b is a numpy.ndarray of shape (1, 1, 1, c_new) containing
      the biases applied to the convolution
    * activation is an activation function applied to the convolution
    * padding is a string that is either same or valid, indicating the
      type of padding used
    * stride is a tuple of (sh, sw) containing the strides for the convolution
         sh is the stride for the height
         sw is the stride for the width
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(((h_prev - 1) * sh + kh - h_prev) / 2) + 1
        pw = int(((w_prev - 1) * sw + kw - w_prev) / 2) + 1

    else:
        ph, pw = (0, 0)

    # applying padding
    new_prev = np.pad(A_prev,
                      ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                      'constant')
    # dims conv matrix
    conv_h = int(((h_prev + (2 * ph) - kh) / sh) + 1)
    conv_w = int(((w_prev + (2 * pw) - kw) / sw) + 1)
    conv = np.zeros((m, conv_h, conv_w, c_new))

    for i in range(conv_h):
        for j in range(conv_w):
            for f in range(c_new):
                sth = i * sh
                endh = (i * sh) + kh
                stw = j * sw
                endw = (j * sw) + kw
                X = new_prev[:, sth:endh, stw:endw]
                WX = (W[:, :, :, f] * X)
                WX = WX.sum(axis=(1, 2, 3))
                conv[:, i, j, f] = WX

    Z = conv + b
    return activation(Z)
