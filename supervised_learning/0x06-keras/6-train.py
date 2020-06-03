#!/usr/bin/env python3
"""
Train a model with keras and Early Stopping
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    early stopping
    """
    my_list = []

    if validation_data and early_stopping:
        my_list.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                 patience=patience))

    return network.fit(x=data,
                       y=labels,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=validation_data,
                       shuffle=shuffle,
                       callbacks=my_list)
