#!/usr/bin/env python3
"""
Batch Normalization in tensorflow
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be
               used on the output of the layer
    you should use the tf.layers.Dense layer as the base layer with kernal
     initializer tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    your layer should incorporate two trainable parameters, gamma and beta,
      initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-8
    Returns: a tensor of the activated output for the layer
    """
    # layers initialization
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=w_init)
    Z = layers(prev)

    # trainable variables gamma and beta
    gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n]),
                       name='beta', trainable=True)
    epsilon = tf.constant(1e-8)

    # Normalization Process
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)

    # activation of Z obtained None if it is the output layer
    if not activation:
        return Z-norm
    else:
        A = activation(Z_norm)
        return A
