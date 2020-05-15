#!/usr/bin/env python3
"""
calculate Accuracy
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing the networks predictions
    Returns: a tensor containing the decimal
             accuracy of the prediction
    """
    index_y = tf.math.argmax(y, axis=1)
    index_pred = tf.math.argmax(y_pred, axis=1)
    comp = tf.math.equal(index_y, index_pred)
    cast = tf.cast(comp, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)
    return accuracy
