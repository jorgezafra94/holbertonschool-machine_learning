#!/usr/bin/env python3
"""
Creating Transition layer
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    * X is the output from the previous layer
    * nb_filters is an integer representing
        the number of filters in X
    * compression is the compression factor for the transition layer
    * Your code should implement compression as used in DenseNet-C
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization and
        a rectified linear activation (ReLU), respectively
    * Returns: The output of the transition layer and the number of
        filters within the output, respectively
    """
    lay_init = K.initializers.he_normal()
    filter = int(nb_filters * compression)
    norm = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation('relu')(norm)
    conv = K.layers.Conv2D(filters=filter,
                           kernel_size=(1, 1),
                           padding="same", strides=(1, 1),
                           kernel_initializer=lay_init)(act)
    avgpool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(conv)
    return (avgpool, filter)
