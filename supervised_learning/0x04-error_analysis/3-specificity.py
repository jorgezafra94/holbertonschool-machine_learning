#!/usr/bin/env python3
"""
Specificity confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
     * confusion is a confusion numpy.ndarray of shape (classes, classes) where
              row indices represent the correct labels and column indices
              represent the predicted labels
    * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the specificity
              of each class
    """
    truePositive = confusion.diagonal()
    falsePositive = confusion.sum(axis=0) - truePositive
    falseNegative = confusion.sum(axis=1) - truePositive
    aux = truePositive + falsePositive + falseNegative
    trueNegative = confusion.sum() - aux
    specificity = trueNegative / (trueNegative + falsePositive)
    return specificity
