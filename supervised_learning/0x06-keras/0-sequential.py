#!/usr/bin/env python3
"""
Creating Model in keras
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

    # here with Sequential we are going to store the model
    model = K.Sequential()

    # First layer with activation and input_shape(*,nx)
    model.add(K.layers.Dense(units=layers[0], activation=activations[0],
                             kernel_regularizer=reg, input_shape=(nx,)))

    for i in range(1, len(layers)):
        # adding dropout to the previous layer
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(units=layers[i], activation=activations[i],
                                 kernel_regularizer=reg))

    return model
