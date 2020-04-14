#!/usr/bin/env python3
"""
using concatenate method of numpy
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    new = np.concatenate((mat1, mat2), axis=axis)
    return new
