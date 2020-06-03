#!/usr/bin/env python3
"""
gray_filter using padding
"""

import numpy as np

def convolve_grayscale_same(images, kernel):
    """

    Args:
        images:
        kernel:

    Returns:

    """
    HK, WK = kernel.shape
    if (HK % 2 == 0):
        pad_H = HK // 2
    else:
        pad_H = (HK - 1) // 2
    if (WK % 2 == 0):
        pad_W = WK // 2
    else:
        pad_W = (WK - 1) // 2

    new_images = np.pad(images,
                        ((0,0),(pad_H, pad_H),(pad_W, pad_W)),
                        'constant')
    m, HI, WI = new_images.shape
    new_H = (HI - HK + 1)
    new_W = (WI - WK + 1)
    conv = np.zeros((m, new_H, new_W))
    for i in range(new_H):
        for j in range(new_W):
            image_part = new_images[:, i:i + HK, j:j + WK]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv