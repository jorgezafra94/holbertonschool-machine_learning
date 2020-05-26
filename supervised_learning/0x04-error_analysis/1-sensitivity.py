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
    sensitivity = confusion.diagonal()
    totalPerClass = confusion.sum(axis=1)
    sensitivity = sensitivity / totalPerClass
    return sensitivity
