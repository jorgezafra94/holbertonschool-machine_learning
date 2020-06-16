#!/usr/bin/env python3
"""
creating GoogLeNet
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block
        should use a rectified linear activation (ReLU)
    Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    lay_init = K.initializers.he_normal()

    # we have to use same to fulfill the dimensions
    conv_lay1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                                strides=(2, 2), padding='same',
                                activation='relu',
                                kernel_initializer=lay_init)
    conv1 = conv_lay1(X)

    maxp_lay1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool1 = maxp_lay1(conv1)

    conv_lay2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1),
                                activation='relu',
                                kernel_initializer=lay_init)
    conv2 = conv_lay2(maxpool1)

    conv_lay3 = K.layers.Conv2D(filters=192, kernel_size=(3, 3),
                                padding='same', activation='relu',
                                kernel_initializer=lay_init)
    conv3 = conv_lay3(conv2)

    maxp_lay2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool2 = maxp_lay2(conv3)

    inception1 = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    maxp_lay3 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool3 = maxp_lay3(inception2)

    inception3 = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])
    inception5 = inception_block(inception4, [128, 128, 256, 24, 64, 64])
    inception6 = inception_block(inception5, [112, 144, 288, 32, 64, 64])
    inception7 = inception_block(inception6, [256, 160, 320, 32, 128, 128])

    maxp_lay4 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')
    maxpool4 = maxp_lay4(inception7)

    inception8 = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    inception9 = inception_block(inception8, [384, 192, 384, 48, 128, 128])

    avgp_lay = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))
    avgpool = avgp_lay(inception9)

    drop_lay = K.layers.Dropout(0.4)
    dropout = drop_lay(avgpool)

    FC_lay = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=lay_init)
    FC = FC_lay(dropout)

    model = K.models.Model(inputs=X, outputs=FC)
    return model
