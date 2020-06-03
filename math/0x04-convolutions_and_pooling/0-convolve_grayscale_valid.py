#!/usr/bin/env python3
"""
applying filter to image
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Args:
        images:
        kernel:

    Returns:
    """
    m, HI, WI = images.shape
    HK, WK = kernel.shape
    new_H = (HI - HK + 1)
    new_W = (WI - WK + 1)

    conv = np.zeros((m, new_H, new_W))
    for i in range(new_H):
        for j in range(new_W):
            image_part = images[:, i:i+HK, j:j+WK]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv
