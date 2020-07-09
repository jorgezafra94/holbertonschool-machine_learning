#!/usr/bin/env python3
"""
contains the TrainModel class
"""

from triplet_loss import TripletLoss
import tensorflow as tf
import numpy as np


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

        # each one of the A P N have to enter in the base_model model
        predict_a = self.base_model(A)
        predict_b = self.base_model(P)
        predict_c = self.base_model(N)

        # the outputs have to enter in the tl
        tl = TripletLoss(alpha)
        output = tl([predict_a, predict_b, predict_c])

        # In this way we send 3 different inputs
        model_fin = tf.keras.models.Model([A, P, N], outputs=output)
        model_fin.compile(optimizer='adam')
        self.training_model = model_fin

    def train(self, triplets, epochs=5, batch_size=32,
              validation_split=0.3, verbose=True):
        """
        * triplets is a list of numpy.ndarrayscontaining
            the inputs to self.training_model
        * epochs is the number of epochs to train for
        * batch_size is the batch size for training
        * validation_split is the validation split for training
        * verbose is a boolean that sets the verbosity mode
        """
        history = self.training_model.fit(x=triplets,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          validation_split=validation_split,
                                          verbose=verbose)
        return history

    def save(self, save_path):
        """
        * save_path is the path to save the model
        * Returns: the saved model
        """
        tf.keras.models.save_model(self.base_model, save_path)
        return self.base_model

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        * y_true - a numpy.ndarray of shape (m,) containing the correct labels
        * m is the number of examples
        * y_pred- a numpy.ndarray of shape (m,) containing the predicted labels
        Returns: The f1 score
        """
        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)
        sensitivity = TP / (TP + FN)
        precision = TP / (TP + FP)

        f1 = (2 * sensitivity * precision) / (sensitivity + precision)
        return f1

    @staticmethod
    def accuracy(y_true, y_pred):
        """
        * y_true - a numpy.ndarray of shape (m,) containing the correct labels
        * m is the number of examples
        * y_pred- a numpy.ndarray of shape (m,) containing the predicted labels
        Returns: the accuracy
        """
        TP = np.count_nonzero(y_pred * y_true)
        TN = np.count_nonzero((y_pred - 1) * (y_true - 1))
        FP = np.count_nonzero(y_pred * (y_true - 1))
        FN = np.count_nonzero((y_pred - 1) * y_true)

        # accuracy
        acc = (TP + TN) / (TP + TN + FP + FN)
        return acc

    def best_tau(self, images, identities, thresholds):
        """
        * images - a numpy.ndarray of shape (m, n, n, 3) containing the
          aligned images for testing
        * m is the number of images
        * n is the size of the images
        * identities - a list containing the identities of each image
          in images
        * thresholds - a 1D numpy.ndarray of distance thresholds (tau)
          to test
        Returns: (tau, f1, acc)
        * tau- the optimal threshold to maximize F1 score
        * f1 - the maximal F1 score
        * acc - the accuracy associated with the maximal F1 score
        """

        distancias = []
        identicas = []
        # my_face = tf.keras.models.load_model('models/trained_fv.h5')
        # pro_img = my_face.predict(images)
        pro_img = self.base_model.predict(images)

        for i in range(len(identities) - 1):
            for j in range(i + 1, len(identities)):
                dist = (np.square(pro_img[i] - pro_img[j]))
                dist = np.sum(dist)
                print(dist, identities[i], identities[j])
                distancias.append(dist)
                if identities[i] == identities[j]:
                    identicas.append(1)
                else:
                    identicas.append(0)

        distancias = np.array(distancias)
        identicas = np.array(identicas)

        f1_list = []
        acc_list = []

        for t in thresholds:
            mask = np.where(distancias <= t, 1, 0)
            f1 = self.f1_score(identicas, mask)
            acc = self.accuracy(identicas, mask)
            f1_list.append(f1)
            acc_list.append(acc)

        f1_max = max(f1_list)
        index = f1_list.index(f1_max)
        acc_max = acc_list[index]
        tau = thresholds[index]

        return(tau, f1_max, acc_max)
