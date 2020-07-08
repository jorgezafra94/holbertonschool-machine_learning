#!/usr/bin/env python3
"""contains the TrainModel class"""
from triplet_loss import TripletLoss
import tensorflow as tf
class TrainModel:
    """
    TrainModel class
    """
    def __init__(self, model_path, alpha):
        """
        constructor
        :param model_path: path to the base face verification embedding model
            loads the model using with tf.keras.utils.CustomObjectScope({'tf': tf}):
            saves this model as the public instance method base_model
        :param alpha: alpha to use for the triplet loss calculation
        """
        """with tf.keras.utils.CustomObjectScope({'tf': tf}):"""
        self.training_model = tf.keras.models.load_model(model_path)
        """self.base_model.save('base_model')
        A = tf.placeholder(tf.float32, (None, 96, 96, 3))
        P = tf.placeholder(tf.float32, (None, 96, 96, 3))
        N = tf.placeholder(tf.float32, (None, 96, 96, 3))
        inputs = [A, P, N]
        output = self.base_model(inputs)
        tl = TripletLoss(alpha)
        output = tl(output)
        self.training_model = tf.keras.models.Model(inputs, output)
        self.training_model.compile(optimizer='Adam')
        self.training_model.save('training_model')"""








