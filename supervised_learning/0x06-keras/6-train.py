#!/usr/bin/env python3
"""
Train a model with keras and Early Stopping
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
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
    * early_stopping is a boolean that indicates whether early stopping
          should be used
          early stopping should only be performed if validation_data exists
          early stopping should be based on validation loss
    * patience is the patience used for early stopping
    Returns: the History object generated after training the model
    """
    early = None

    if early_stopping is True and validation_data:
        early = [K.callbacks.EarlyStopping(patience=patience)]

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data, callbacks=early)

    return history
