#!/usr/bin/env python3
"""
Save best model
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
    * learning_rate_decay is a boolean that indicates whether learning rate
          decay should be used
      learning rate decay should only be performed if validation_data exists
      the decay should be performed using inverse time decay
      the learning rate should decay in a stepwise fashion after each epoch
    * each time the learning rate updates, Keras should print a message
           alpha is the initial learning rate
           decay_rate is the decay rate
    * save_best is a boolean indicating whether to save the model after each
           epoch if it is the best
    * a model is considered the best if its validation loss is the lowest that
           the model has obtained
    * filepath is the file path where the model should be saved
    Returns: the History object generated after training the model
    """

    def learning_decay(epoch):
        """
        funcion in the learningRateSchedule
        the alpha doesnt change
        """
        return alpha / (1 + decay_rate * (epoch / 1))

    my_list = []
    if early_stopping is True and validation_data:
        early = K.callbacks.EarlyStopping(patience=patience)
        my_list.append(early)

    if learning_rate_decay is True and validation_data:
        learn_dec = K.callbacks.LearningRateScheduler(learning_decay,
                                                      verbose=1)
        my_list.append(learn_dec)

    if save_best is True and validation_data:
        save = K.callbacks.ModelCheckpoint(filepath=filepath,
                                           save_best_only=True)
        my_list.append(save)

    if len(my_list) == 0:
        my_list = None

    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data, callbacks=my_list)
    return history
