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
    m, ih, iw = images.shape
    kh, kw = kernel.shape
    new_H = (ih - kh + 1)
    new_W = (ih - kw + 1)

    conv = np.zeros((m, new_H, new_W))
    for i in range(new_H):
        for j in range(new_W):
            # we get here same the part for each one of the images
            image_part = images[:, i:i+kh, j:j+kw]
            # we get here the result of multiplication
            result = image_part * kernel
            # we have to get a number from result (matrix of matrices)
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            # once we get the numbers we are going to
            # store the numbers in the same position for
            # each image
            conv[:, i, j] = result
            # every time we go through the loop we are getting
            # one position of the conv for each one of the images
    return conv
