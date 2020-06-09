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
    _, h_prev, w_prev, _ = A_prev.shape
    m, h_new, w_new, c_new = dZ.shape
    kh, kw, c_prev, _ = W.shape
    ph, pw = (0, 0)
    sh, sw = stride

    if padding == "same":
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2) + 1)
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2) + 1)

    # initialize the derivatives
    dA = np.zeros((m, h_prev, w_prev, c_prev))
    dW = np.zeros((kh, kw, c_prev, c_new))
    db = np.zeros((1, 1, 1, c_new))

    A_pad = np.pad(A_prev,
                   ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   'constant')

    dA_pad = np.pad(dA,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant')
    for elem in range(m):
        im = A_pad[elem]
        dIm = dA_pad[elem]
        for i in range(h_new):
            for j in range(w_new):
                for f in range(c_new):
                    sth = i * sh
                    endh = (i * sh) + kh
                    stw = j * sw
                    endw = (j * sw) + kw
                    X = im[sth:endh, stw:endw]
                    # to get the back part of the image --> dim += W * dZ
                    # it is important to use the + simbol careful with this
                    aux = W[:, :, :, f] * dZ[elem, i, j, f]
                    dIm[sth:endh, stw:endw] += aux

                    # to get the backprop of the W --> dw += dz * x
                    dW[:, :, :, f] += X * dZ[elem, i, j, f]

                    # to get the backprop of b ----> db += dz
                    db[:, :, :, f] += dZ[elem, i, j, f]
        if (padding == 'valid'):
            dA[elem] = dIm
        if (padding == 'same'):
            dA[elem] = dIm[ph: -ph, pw: -pw]

    return (dA, dW, db)
