#!/usr/bin/env python3
"""
lenet5 with keras
"""

import tensorflow.keras as K


def lenet5(X):
    """
    * X is a K.Input of shape (m, 28, 28, 1) containing the
      input images for the network
          m is the number of images
    * The model should consist of the following layers in order:
         - Convolutional layer with 6 kernels of shape 5x5 with same padding
         - Max pooling layer with kernels of shape 2x2 with 2x2 strides
         - Convolutional layer with 16 kernels of shape 5x5 with valid padding
         - Max pooling layer with kernels of shape 2x2 with 2x2 strides
         - Fully connected layer with 120 nodes
         - Fully connected layer with 84 nodes
         - Fully connected softmax output layer with 10 nodes
         - All layers requiring initialization should initialize their
           kernels with the he_normal initialization method
         - All hidden layers requiring activation should use the relu
           activation function
    * you may import tensorflow.keras as K
    Returns: a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    lay_init = K.initializers.he_normal(seed=None)
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv_layer1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                  padding='same', activation='relu',
                                  kernel_initializer=lay_init)
    conv1 = conv_layer1(X)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool_layer1 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool1 = pool_layer1(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv_layer2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                  padding='valid', activation='relu',
                                  kernel_initializer=lay_init)
    conv2 = conv_layer2(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool_layer2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool2 = pool_layer2(conv2)

    # change form matrix to vector
    x_vec = K.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    layer3 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=lay_init)
    FC3 = layer3(x_vec)

    # Fully connected layer with 84 nodes
    layer4 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=lay_init)
    FC4 = layer4(FC3)

    # Fully connected softmax output layer with 10 nodes
    layer5 = K.layers.Dense(units=10, activation='softmax',
                            kernel_initializer=lay_init)
    y_pred = layer5(FC4)

    # model structure
    model = K.models.Model(inputs=X, outputs=y_pred)
    # compile model
    Adam = K.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=Adam,
                  metrics=['accuracy'])
    return model
