#!/usr/bin/env python3
"""
creating lenet5 architecture in Tensorflow
"""

import tensorflow as tf


def lenet5(x, y):
    """
    * x is a tf.placeholder of shape (m, 28, 28, 1) containing
      the input images for the network
          m is the number of images
    * y is a tf.placeholder of shape (m, 10) containing the one-hot
      labels for the network
    * The model should consist of the following layers in order:
        - Convolutional layer with 6 kernels of shape 5x5 with same padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Convolutional layer with 16 kernels of shape 5x5 with valid padding
        - Max pooling layer with kernels of shape 2x2 with 2x2 strides
        - Fully connected layer with 120 nodes
        - Fully connected layer with 84 nodes
        - Fully connected softmax output layer with 10 nodes
        - All layers requiring initialization should initialize their kernels
          with the he_normal initialization
          method: tf.contrib.layers.variance_scaling_initializer()
        - All hidden layers requiring activation should use the relu
          activation function
        - you may import tensorflow as tf
        - you may NOT use tf.keras
    Returns:
    - a tensor for the softmax activated output
    - a training operation that utilizes Adam
      optimization (with default hyperparameters)
    - a tensor for the loss of the netowrk
    - a tensor for the accuracy of the network
    """
    lay_init = tf.contrib.layers.variance_scaling_initializer()
    act_f = tf.nn.relu
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv_layer1 = tf.layers.Conv2D(filters=6, kernel_size=(5, 5),
                                   padding='same', activation=act_f,
                                   kernel_initializer=lay_init)
    conv1 = conv_layer1(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool_layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool1 = pool_layer1(conv1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv_layer2 = tf.layers.Conv2D(filters=16, kernel_size=(5, 5),
                                   padding='valid', activation=act_f,
                                   kernel_initializer=lay_init)
    conv2 = conv_layer2(pool1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    pool_layer2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    pool2 = pool_layer2(conv2)

    # change form matrix to vector
    x_vec = tf.layers.Flatten()(pool2)

    # Fully connected layer with 120 nodes
    layer3 = tf.layers.Dense(units=120, activation=act_f,
                             kernel_initializer=lay_init)
    FC3 = layer3(x_vec)

    # Fully connected layer with 84 nodes
    layer4 = tf.layers.Dense(units=84, activation=act_f,
                             kernel_initializer=lay_init)
    FC4 = layer4(FC3)

    # Fully connected softmax output layer with 10 nodes
    layer5 = tf.layers.Dense(units=10, kernel_initializer=lay_init)
    y_pred = layer5(FC4)
    # loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # accuracy
    index_y = tf.math.argmax(y, axis=1)
    index_pred = tf.math.argmax(y_pred, axis=1)
    comp = tf.math.equal(index_y, index_pred)
    cast = tf.cast(comp, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)

    # train with Adam optimizer
    Adam = tf.train.AdamOptimizer().minimize(loss)

    y_softmax = tf.nn.softmax(y_pred)
    return (y_softmax, Adam, loss, accuracy)
