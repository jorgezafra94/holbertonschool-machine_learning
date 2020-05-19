#!/usr/bin/env python3
"""
gradient descent with momentum in TensorFlow
"""

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    Returns: the momentum optimization operation
    """
    a = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    optimization = a.minimize(loss=loss)
    return optimization
