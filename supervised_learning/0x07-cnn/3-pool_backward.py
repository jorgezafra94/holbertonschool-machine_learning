#!/usr/bin/env python3
"""
backpropagation using maxpooling
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    * dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
      the partial derivatives with respect to the output of the pooling layer
          m is the number of examples
          h_new is the height of the output
          w_new is the width of the output
          c is the number of channels
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
      the output of the previous layer
          h_prev is the height of the previous layer
          w_prev is the width of the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size of the kernel
      for the pooling
          kh is the kernel height
          kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
          sh is the stride for the height
          sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether
      to perform maximum or average pooling, respectively
    Returns: the partial derivatives with respect to the
    previous layer (dA_prev)
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # creating derivatives
    dA_prev = np.zeros(A_prev.shape)

    for elem in range(m):
        im = A_prev[elem]
        for i in range(h_new):
            for j in range(w_new):
                for f in range(c_new):
                    sth = i * sh
                    endh = (i * sh) + kh
                    stw = j * sw
                    endw = (j * sw) + kw
                    X = im[sth:endh, stw:endw, f]
                    if mode == 'max':
                        mask = np.where(X == np.max(X), 1, 0)
                    if mode == 'avg':
                        mask = np.ones(X.shape)
                    aux = mask * dA[elem, i, j, f]
                    dA_prev[elem, sth:endh, stw:endw, f] = aux
    return dA_prev
