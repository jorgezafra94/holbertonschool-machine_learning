#!/usr/bin/env python3
"""
Creating Dense Block
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    * X is the output from the previous layer
    * nb_filters is an integer representing the
        number of filters in X
    * growth_rate is the growth rate for the dense block
    * layers is the number of layers in the dense block
    * You should use the bottleneck layers used for DenseNet-B
    * All weights should use he normal initialization
    * All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively
    * Returns: The concatenated output of each layer within the
        Dense Block and the number of filters within the
        concatenated outputs, respectively
    """
    lay_init = K.initializers.he_normal()

    for i in range(layers):
        norm1 = K.layers.BatchNormalization(axis=3)(X)
        act1 = K.layers.Activation('relu')(norm1)
        bottle = K.layers.Conv2D(filters=(4 * growth_rate),
                                 kernel_size=(1, 1),
                                 padding="same", strides=(1, 1),
                                 kernel_initializer=lay_init)(act1)
        norm2 = K.layers.BatchNormalization(axis=3)(bottle)
        act2 = K.layers.Activation('relu')(norm2)
        conv3 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding="same", strides=(1, 1),
                                kernel_initializer=lay_init)(act2)
        X = K.layers.concatenate([X, conv3])
        nb_filters += growth_rate
    return (X, nb_filters)
