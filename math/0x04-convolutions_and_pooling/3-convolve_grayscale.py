#!/usr/bin/env python3
"""
gray_filter same=padding, valid=no padding, stride=pace
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images is a numpy.ndarray with shape (m, h, w) containing multiple
    grayscale images
          m is the number of images
          h is the height in pixels of the images
          w is the width in pixels of the images
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
    for the convolution
          kh is the height of the kernel
          kw is the width of the kernel
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
    Returns: a numpy.ndarray containing the convolved images

    """
    kh, kw = kernel.shape
    sh, sw = stride
    m, ih, iw = images.shape

    # padding part
    if type(padding) is tuple and len(padding) == 2:
        ph, pw = padding

    elif padding == 'same':
        if kh % 2 == 0:
            ph = kh // 2
        else:
            ph = (kh - 1) // 2

        if kw % 2 == 0:
            pw = kw // 2
        else:
            pw = (kw - 1) / 2

    else:
        ph, pw = 0, 0

    # applying padding in input images
    new_images = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw)),
                        'constant')

    # when we apply stride the ecuations change
    new_H = (((ih + (2 * ph) - kh) // sh) + 1)
    new_W = (((iw + (2 * pw) - kw) // sw) + 1)

    if padding == 'same':
        conv = np.zeros((m, ih, iw))
        new_H = ih
        new_W = iw
    else:
        conv = np.zeros((m, new_H, new_W))

    for i in range(0, new_H):
        for j in range(0, new_W):
            # we have to change the pace because the stride
            image_part = new_images[:, (i * sh):(sh * i) + kh,
                                    (j * sw):(sw * j) + kw]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv
