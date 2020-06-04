#!/usr/bin/env python3
"""
applying stride, padding and with several filter to RGB pictures
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w, c) containing
    multiple images
           m is the number of images
           h is the height in pixels of the images
           w is the width in pixels of the images
           c is the number of channels in the image
    kernel is a numpy.ndarray with shape(kh, kw, c, nc) containing the kernel
    for the convolution
           kh is the height of the kernel
           kw is the width of the kernel
           nc is the number of kernels
    padding is either a tuple of (ph, pw), same, or valid
        if same, performs a same convolution
        if valid, performs a valid convolution
        if a tuple:
           ph is the padding for the height of the image
           pw is the padding for the width of the image
    the image should be padded with 0s
    stride is a tuple of (sh, sw)
          sh is the stride for the height of the image
          sw is the stride for the width of the image
    """
    # getting information
    m, ih, iw, ic = images.shape
    kh, kw, _, kc = kernels.shape
    sh, sw = stride

    # getting padding
    if padding == 'same':
        if kh % 2 == 0:
            ph = kh // 2
        else:
            ph = (kh - 1) // 2
        if kw % 2 == 0:
            pw = kw // 2
        else:
            pw = (kw - 1) // 2

    if padding == 'valid':
        ph, pw = (0, 0)

    if type(padding) is tuple and len(padding) == 2:
        ph, pw = padding

    # applying padding to images RGB
    new_images = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        'constant')

    # getting shape of convolutional matrix
    new_h = ((ih + (2 * ph) - kh) // sh) + 1
    new_w = ((iw + (2 * pw) - kw) // sw) + 1
    conv = np.zeros((m, new_h, new_w, kc))

    # we have to add a new for to apply each filter
    for fil in range(kc):
        for i in range(new_h):
            for j in range(new_w):
                part = new_images[:, (i * sh): (i * sh) + kh,
                                  (j * sw): (j * sw) + kw]
                # here we get the new matrix of matrices
                # and we choose the filter to apply
                result = part * kernels[:, :, :, fil]
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                result = result.sum(axis=1)
                conv[:, i, j, fil] = result

    return conv
