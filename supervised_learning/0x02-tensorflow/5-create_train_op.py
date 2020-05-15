#!/usr/bin/env python3
"""
Train function Tensorflow
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    loss is the loss of the networks prediction
    alpha is the learning rate
    Returns: an operation that trains the
             network using gradient descent
    """
    result = tf.train.GradientDescentOptimizer(alpha)
    final = result.minimize(loss)
    return final
