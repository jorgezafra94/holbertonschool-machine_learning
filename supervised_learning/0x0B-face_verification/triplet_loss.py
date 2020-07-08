#!/usr/bin/env python3
"""
Triple Loss
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Layer


class TripletLoss(Layer):
    """
    class Tripletloss
    """

    def __init__(self, alpha, **kwargs):
        """
        * alpha is the alpha value used to calculate the triplet loss
        * sets the public instance attribute alpha
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def triplet_loss(self, inputs):
        """
        * inputs is a list containing the anchor, positive and negative
            output tensors from the last layer of the model, respectively
        Returns: a tensor containing the triplet loss values
        """
        A, P, N = inputs
        # ********************** Keras ****************************
        # pos_dist = K.backend.sum(K.backend.square(A - P), axis=-1)
        # neg_dist = K.backend.sum(K.backend.square(A - N), axis=-1)
        # basic_loss = K.layers.Subtract()([pos_dist, neg_dist]) + self.alpha
        # loss = K.backend.maximum(basic_loss, 0)

        # ********************* Tensorflow ******************************
        pos_dist = tf.reduce_sum((A - P) ** 2, axis=-1)
        neg_dist = tf.reduce_sum((A - N) ** 2, axis=-1)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.maximum(basic_loss, 0)

        return loss

    def call(self, inputs):
        """
        * inputs is a list containing the anchor, positive, and negative
            output tensors from the last layer of the model, respectively
        * adds the triplet loss to the graph
        Returns: the triplet loss tensor
        """
        #  you can create loss tensors that you will want to use later
        # this let us getting model.losses and tl._losses
        self.add_loss(self.triplet_loss(inputs))
        return self.triplet_loss(inputs)
