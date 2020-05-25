#!/usr/bin/env python3
"""
Precision confusion matrix
"""


import numpy as np


def precision(confusion):
    """
     * confusion is a confusion numpy.ndarray of shape (classes, classes) where
              row indices represent the correct labels and column indices
              represent the predicted labels
    * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the precision
            of each class
    """
    presicion = np.zeros(confusion.shape[0])
    totalPerClassPred = confusion.sum(axis=0)
    for i in range(confusion.shape[0]):
        presicion[i] = confusion[i][i] / totalPerClassPred[i]
    return presicion
