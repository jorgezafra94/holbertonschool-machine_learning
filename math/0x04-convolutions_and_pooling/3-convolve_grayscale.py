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
    HK, WK = kernel.shape
    sh, sw = stride
    if type(padding) is tuple and len(padding) == 2:
        pad_H, pad_W = padding

    elif padding == 'same':
        if HK % 2 == 0:
            pad_H = HK // 2
        else:
            pad_H = (HK - 1) // 2

        if WK % 2 == 0:
            pad_W = WK // 2
        else:
            pad_W = (WK - 1) / 2

    else:
        pad_H, pad_W = 0, 0

    new_images = np.pad(images,
                        ((0, 0), (pad_H, pad_H), (pad_W, pad_W)),
                        'constant')

    m, HI, WI = images.shape
    new_H = (((HI + (2 * pad_H) - HK) // stride[0]) + 1)
    new_W = (((WI + (2 * pad_W) - WK) // stride[1]) + 1)

    conv = np.zeros((m, new_H, new_W))

    for i in range(0, new_H):
        for j in range(0, new_W):
            image_part = new_images[:, i * sh:sh * i + HK, j * sw:sw * j + WK]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv
