#!/usr/bin/env python3
"""
confusion Matrix or error Matrix
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """
    * labels is a one-hot numpy.ndarray of shape (m, classes)
               containing the correct labels for each data point
    * m is the number of data points
    * classes is the number of classes
    * logits is a one-hot numpy.ndarray of shape (m, classes)
               containing the predicted labels
    * Returns: a confusion numpy.ndarray of shape (classes, classes) with row
               indices representing the correct labels and column indices
               representing the predicted labels
    """
    classes = labels.shape[1]
    # Actual
    actual = np.argmax(labels, axis=1)
    # predicted
    pred = np.argmax(logits, axis=1)
    # confusion matrix initialization and creation
    confusion = np.array(np.zeros(classes ** 2))
    confusion = confusion.reshape(classes, classes)
    # fulling the matrix initialization
    np.add.at(confusion, (actual, pred), 1)
    return confusion
