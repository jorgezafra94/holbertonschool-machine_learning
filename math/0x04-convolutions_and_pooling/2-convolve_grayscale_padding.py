#!/usr/bin/env python3
"""
gray_filter using padding
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    padding is a tuple of (ph, pw)
         ph is the padding for the height of the image
         pw is the padding for the width of the image
    the image should be padded with 0s
    """
    kh, kw = kernel.shape
    ph, pw = padding
    new_images = np.pad(images,
                        ((0, 0), (ph, ph), (pw, pw)),
                        'constant')
    m, ih, iw = images.shape
    new_H = (ih + (2 * ph) - kh + 1)
    new_W = (iw + (2 * pw) - kw + 1)
    conv = np.zeros((m, new_H, new_W))
    for i in range(new_H):
        for j in range(new_W):
            image_part = new_images[:, i:i + kh, j:j + kw]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv
