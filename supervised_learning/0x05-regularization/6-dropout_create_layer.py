#!/usr/bin/env python3
"""
Dropout with TensorFlow
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    * prev is a tensor containing the output of the previous layer
    * n is the number of nodes the new layer should contain
    * activation is the activation function that should be used on the layer
    * keep_prob is the probability that a node will be kept
    * Returns: the output of the new layer
    """
    init_w = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dropout = tf.layers.Dropout(rate=keep_prob)
    layers = tf.layers.Dense(units=n, activation=activation,
                             kernel_initializer=init_w,
                             kernel_regularizer=dropout)
    A = layers(prev)

    return A
