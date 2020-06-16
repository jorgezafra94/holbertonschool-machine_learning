#!/usr/bin/env python3
"""
creating Identity block
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    * A_prev is the output from the previous layer
    * filters is a tuple or list containing F11, F3,
        F12, respectively:
    * F11 is the number of filters in the first 1x1
        convolution
    * F3 is the number of filters in the 3x3 convolution
    * F12 is the number of filters in the second 1x1
        convolution
    * All convolutions inside the block should be followed
        by batch normalization along the channels axis and
        a rectified linear activation (ReLU), respectively.
    * All weights should use he normal initialization
    Returns: the activated output of the identity block
    """
    F11, F3, F12 = filters
    lay_init = K.initializers.he_normal()

    # first conv 1x1 with batch normalization
    conv_layF11 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1),
                                  padding='same',
                                  kernel_initializer=lay_init)
    convF11 = conv_layF11(A_prev)

    norm_lay1 = K.layers.BatchNormalization(axis=3)
    norm1 = norm_lay1(convF11)

    X1 = K.layers.Activation('relu')(norm1)

    # conv 3x3 with batch_normalization
    conv_layF3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                                 padding='same',
                                 kernel_initializer=lay_init)
    convF3 = conv_layF3(X1)

    norm_lay2 = K.layers.BatchNormalization(axis=3)
    norm2 = norm_lay2(convF3)

    X2 = K.layers.Activation('relu')(norm2)

    # conv 1x1 with batch_normalization after adding
    conv_layF12 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1),
                                  padding='same',
                                  kernel_initializer=lay_init)
    convF12 = conv_layF12(X2)

    norm_lay3 = K.layers.BatchNormalization(axis=3)
    norm3 = norm_lay3(convF12)

    result = K.layers.Add()([norm3, A_prev])
    X3 = K.layers.Activation('relu')(result)

    return X3
