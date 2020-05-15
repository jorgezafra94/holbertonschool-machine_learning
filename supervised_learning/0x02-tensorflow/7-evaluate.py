#!/usr/bin/env python3
"""
Restore save trained DNN
"""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    You are not allowed to use tf.saved_model
    Returns: the networks prediction, accuracy, and loss,
             respectively
    """
    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(ses, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        out = ses.run(y_pred, feed_dict={x: X, y: Y})
        acc = ses.run(accuracy, feed_dict={x: X, y: Y})
        cost = ses.run(loss, feed_dict={x: X, y: Y})

    return (out, acc, cost)
