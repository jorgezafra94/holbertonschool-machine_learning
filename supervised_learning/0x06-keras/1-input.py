#!/usr/bin/env python3
"""
Creating Model in keras using Input
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    * nx is the number of input features to the network
    * layers is a list containing the number of nodes in each layer of the
        network
    * activations is a list containing the activation functions used for each
        layer of the network
    * lambtha is the L2 regularization parameter
    * keep_prob is the probability that a node will be kept for dropout
    * You are not allowed to use the Input class
    * Returns: the keras model
    """
    reg = K.regularizers.l2(lambtha)

    X = K.Input(shape=(nx,))
    layer_l2 = K.layers.Dense(units=layers[0], activation=activations[0],
                              kernel_regularizer=reg)
    Y_prev = layer_l2(X)

    for i in range(1, len(layers)):

        layer_drop = K.layers.Dropout(1 - keep_prob)
        Y = layer_drop(Y_prev)

        layer_l2 = K.layers.Dense(units=layers[i], activation=activations[i],
                                  kernel_regularizer=reg)
        Y_prev = layer_l2(Y)

    Y_pred = Y_prev
    model = K.Model(inputs=X, outputs=Y_pred)
    return model
