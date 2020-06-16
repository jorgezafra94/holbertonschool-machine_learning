#!/usr/bin/env python3
"""
using Inception layer with reduced dimensions
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    * A_prev is the output from the previous layer
    * filters is a tuple or list containing F1, F3R,
        F3,F5R, F5, FPP, respectively:
    * F1 is the number of filters in the 1x1 convolution
    * F3R is the number of filters in the 1x1 convolution
        before the 3x3 convolution
    * F3 is the number of filters in the 3x3 convolution
    * F5R is the number of filters in the 1x1 convolution
        before the 5x5 convolution
    * F5 is the number of filters in the 5x5 convolution
    * FPP is the number of filters in the 1x1 convolution
        after the max pooling
    * All convolutions inside the inception block should
        use a rectified linear activation (ReLU)
    * Returns: the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    lay_init = K.initializers.he_normal()
    convF1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1), padding='same',
                             activation='relu', kernel_initializer=lay_init)

    convF3R = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)

    convF3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                             activation='relu', kernel_initializer=lay_init)

    convF5R = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)

    convF5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5), padding='same',
                             activation='relu', kernel_initializer=lay_init)

    maxpool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                    padding='same')

    convFPP = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1), padding='same',
                              activation='relu', kernel_initializer=lay_init)
    xF1 = convF1(A_prev)
    xF3 = convF3(convF3R(A_prev))
    xF5 = convF5(convF5R(A_prev))
    xFPP = convFPP(maxpool(A_prev))

    result = K.layers.concatenate([xF1, xF3, xF5, xFPP])

    return result
