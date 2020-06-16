#!/usr/bin/env python3
"""
Creating DenseNet
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    * growth_rate is the growth rate
    * compression is the compression factor
    * You can assume the input data will have shape (224, 224, 3)
    * All convolutions should be preceded by Batch Normalization
        and a rectified linear activation (ReLU), respectively
    * All weights should use he normal initialization
    * Returns: the keras model
    """
    X = K.Input(shape=(224, 224, 3))
    lay_init = K.initializers.he_normal()

    norm1 = K.layers.BatchNormalization(axis=3)(X)
    act1 = K.layers.Activation('relu')(norm1)
    conv1 = K.layers.Conv2D(filters=(2 * growth_rate),
                            kernel_size=(7, 7),
                            padding="same", strides=(2, 2),
                            kernel_initializer=lay_init)(act1)
    mxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(2, 2),
                                    padding='same')(conv1)

    dense1, fil_d1 = dense_block(mxpool1, (2 * growth_rate), growth_rate, 6)

    transition1, fil_t1 = transition_layer(dense1, fil_d1, compression)

    dense2, fil_d2 = dense_block(transition1, fil_t1, growth_rate, 12)

    transition2, fil_t2 = transition_layer(dense2, fil_d2, compression)

    dense3, fil_d3 = dense_block(transition2, fil_t2, growth_rate, 24)

    transition3, fil_t3 = transition_layer(dense3, fil_d3, compression)

    dense4, fil_d4 = dense_block(transition3, fil_t3, growth_rate, 16)

    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1))(dense4)

    FC = K.layers.Dense(units=1000, activation='softmax',
                        kernel_initializer=lay_init)(avgpool)

    model = K.models.Model(inputs=X, outputs=FC)

    return model
