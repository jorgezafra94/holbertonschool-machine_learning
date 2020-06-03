#!/usr/bin/env python3
"""
Train a model with keras
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    * network is the model to train
    * data is a numpy.ndarray of shape (m, nx) containing the input data
    * labels is a one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data
    * batch_size is the size of the batch used for mini-batch gradient descent
    * epochs is the number of passes through data for mini-batch
             gradient descent
    * verbose is a boolean that determines if output should be printed during
            training
    * shuffle is a boolean that determines whether to shuffle the batches every
          epoch. Normally, it is a good idea to shuffle, but for
          reproducibility we have chosen to set the default to False.
    Returns: the History object generated after training the model
    """
    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle)
