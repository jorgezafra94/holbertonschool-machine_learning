#!/usr/bine/env python3
"""
slice like a ninja
"""


import numpy as np


def np_slice(matrix, axes={}):
    listSlice = []
    for i in range(len(matrix.shape)):
        flag = 0
        for key in axes:
            if(i == key):
                flag = 1
                listSlice.append(slice(*axes[key]))
        if(flag == 0):
            listSlice.append(slice(None, None, None))
    tupleSlice = tuple(listSlice)
    new = matrix[tupleSlice]
    return new
