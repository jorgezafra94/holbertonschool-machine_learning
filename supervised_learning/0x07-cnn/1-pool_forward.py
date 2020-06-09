#!/usr/bin/env python3
"""
Forward Propagation using maxpooling
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    * A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
      containing the output of the previous layer
          m is the number of examples
          h_prev is the height of the previous layer
          w_prev is the width of the previous layer
          c_prev is the number of channels in the previous layer
    * kernel_shape is a tuple of (kh, kw) containing the size of
      the kernel for the pooling
          kh is the kernel height
          kw is the kernel width
    * stride is a tuple of (sh, sw) containing the strides for the pooling
          sh is the stride for the height
          sw is the stride for the width
    * mode is a string containing either max or avg, indicating whether
      to perform maximum or average pooling, respectively
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # dims conv matrix
    pool_h = int(((h_prev - kh) / sh) + 1)
    pool_w = int(((w_prev - kw) / sw) + 1)
    pool = np.zeros((m, pool_h, pool_w, c_prev))

    for i in range(pool_h):
        for j in range(pool_w):
            sth = i * sh
            endh = (i * sh) + kh
            stw = j * sw
            endw = (j * sw) + kw
            X = A_prev[:, sth:endh, stw:endw]
            if mode == "max":
                WX = np.max(X, axis=(1, 2))
            if mode == "avg":
                WX = np.mean(X, axis=(1, 2))
            pool[:, i, j] = WX
    return pool
