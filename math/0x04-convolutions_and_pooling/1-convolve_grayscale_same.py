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
    kh, kw = kernel.shape
    # when we have to give a padding we have to calculate it like this
    if kh % 2 == 0:
        ph = int(kh / 2)
    else:
        ph = int((kh - 1) / 2)
    if kw % 2 == 0:
        pw = int(kw / 2)
    else:
        pw = int((kw - 1) / 2)

    # We have to apply the padding to the input images
    new_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')
    m, ih, iw = images.shape

    # we get the new shape of the conv matrix using these equations
    # new_H = (ih + (2 * ph) - kh + 1)
    # new_W = (iw + (2 * pw) - kw + 1)

    conv = np.zeros((m, ih, iw))

    for i in range(ih):
        for j in range(iw):
            image_part = new_images[:, i:i + kh, j:j + kw]
            result = image_part * kernel
            result = result.sum(axis=1)
            result = result.sum(axis=1)
            conv[:, i, j] = result
    return conv
