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
