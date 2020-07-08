#!/usr/bin/env python3
"""
contains the TrainModel class
"""

from triplet_loss import TripletLoss
import tensorflow as tf


class TrainModel:
    """
    TrainModel class
    """
    def __init__(self, model_path, alpha):
        """
        * model_path is the path to the base face verification
          embedding model
        * loads the model using with
          tf.keras.utils.CustomObjectScope({'tf': tf}):
        * saves this model as the public instance method base_model
        * alpha is the alpha to use for the triplet loss calculation
        * Creates a new model:
        * * inputs: [A, P, N]
            - A is a numpy.ndarray of shape (m, n, n, 3)containing the
              aligned anchor images
            - P is a numpy.ndarray of shape (m, n, n, 3) containing the
              aligned positive images
            - N is a numpy.ndarray of shape (m, n, n, 3)containing the
              aligned negative images
            - m is the number of images
            - n is the size of the aligned images
        * * outputs: the triplet losses of base_model
        * * compiles the model with:
            - Adam optimization
            - no additional losses
        * * save this model as the public instance attribute training_model
        * you can use from triplet_loss import TripletLoss
        """
        with tf.keras.utils.CustomObjectScope({'tf': tf}):
            self.base_model = tf.keras.models.load_model(model_path)

        # complete Model
        # we create this Inputs because there are the inputs
        # of base_model model
        A = tf.keras.Input(shape=(96, 96, 3))
        P = tf.keras.Input(shape=(96, 96, 3))
        N = tf.keras.Input(shape=(96, 96, 3))
        predict_a = self.base_model(A)
        predict_b = self.base_model(P)
        predict_c = self.base_model(N)
        tl = TripletLoss(alpha)
        output = tl([predict_a, predict_b, predict_c])
        model_fin = tf.keras.models.Model([A, P, N], outputs=output)
        model_fin.compile(optimizer='adam')
        self.training_model = model_fin










