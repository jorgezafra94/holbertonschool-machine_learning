#!/usr/bin/env python3
"""
applying stride, padding and filter to RGB picture
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
         m is the number of images
         h is the height in pixels of the images
         w is the width in pixels of the images
         c is the number of channels in the image
    kernel_shape is a tuple of (kh, kw) containing the kernel shape
    for the pooling
         kh is the height of the kernel
         kw is the width of the kernel
    stride is a tuple of (sh, sw)
         sh is the stride for the height of the image
         sw is the stride for the width of the image
    mode indicates the type of pooling
         max indicates max pooling
         avg indicates average pooling
    """
    # getting information
    m, ih, iw, ic = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # getting shape of convolutional matrix
    new_h = int(((ih - kh) / sh) + 1)
    new_w = int(((iw - kw) / sw) + 1)
    conv = np.zeros((m, new_h, new_w, ic))

    for i in range(new_h):
        for j in range(new_w):
            part = images[:, (i * sh): (i * sh) + kh,
                          (j * sw): (j * sw) + kw]
            # here we get the new matrix of matrices
            if mode == 'max':
                result = np.max(part, axis=1)
                result = np.max(result, axis=1)
            if mode == 'avg':
                result = np.mean(part, axis=1)
                result = np.mean(result, axis=1)
            conv[:, i, j] = result
            print(result.shape)
    return conv
