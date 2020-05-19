#!/usr/bin/env python3
"""
RMSprop optimization algorithm with TensorFlow
"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    Returns: the RMSProp optimization operation
    """
    a = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                  decay=beta2, epsilon=epsilon)
    optimization = a.minimize(loss)
    return optimization
