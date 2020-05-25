#!/usr/bin/env python3
"""
Sensitivity of Confusion matrix
"""


import numpy as np


def sensitivity(confusion):
    """
    * confusion is a confusion numpy.ndarray of shape (classes, classes) where
              row indices represent the correct labels and column indices
              represent the predicted labels
    * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
             of each class
    """
    sensitivity = np.zeros(confusion.shape[0])
    totalPerClass = confusion.sum(axis=1)
    for i in range(confusion.shape[0]):
        sensitivity[i] = confusion[i][i] / totalPerClass[i]
    return sensitivity
