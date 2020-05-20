#!/usr/bin/env python3
"""
Mini Batch
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    X_train is a numpy.ndarray of shape (m, 784) containing
            the training data
    Y_train is a one-hot numpy.ndarray of shape (m, 10)
            containing the training labels
    X_valid is a numpy.ndarray of shape (m, 784) containing
            the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing
            the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should pass
           through the whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be saved
           after training
    """
    with tf.Session() as ses:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(ses, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        train_op = tf.get_collection('train_op')[0]

        mini_iter = X_train.shape[0]/batch_size

        if (mini_iter).is_integer() is True:
            mini_iter = int(mini_iter)
        else:
            mini_iter = (int(mini_iter) + 1)

        train = {x: X_train, y: Y_train}
        valid = {x: X_valid, y: Y_valid}

        for i in range(epochs + 1):
            T_cost = ses.run(loss, feed_dict=train)
            T_acc = ses.run(accuracy, feed_dict=train)
            V_cost = ses.run(loss, feed_dict=valid)
            V_acc = ses.run(accuracy, feed_dict=valid)
            print('After {} epochs:'.format(i))
            print('\tTraining Cost: {}'.format(T_cost))
            print('\tTraining Accuracy: {}'.format(T_acc))
            print('\tValidation Cost: {}'.format(V_cost))
            print('\tValidation Accuracy: {}'.format(V_acc))

            if i < epochs:
                X_shu, Y_shu = shuffle_data(X_train, Y_train)

                for j in range(mini_iter):
                    ini = j * batch_size
                    fin = (j + 1) * batch_size
                    if fin > X_train.shape[0]:
                        fin = X_train.shape[0]
                    new = {x: X_shu[ini:fin], y: Y_shu[ini:fin]}
                    ses.run(train_op, feed_dict=new)
                    if j != 0 and (j + 1) % 100 == 0:
                        Min_cost = ses.run(loss, feed_dict=new)
                        Min_acc = ses.run(accuracy, feed_dict=new)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(Min_cost))
                        print("\t\tAccuracy: {}".format(Min_acc))

        save_path = saver.save(ses, save_path)
    return save_path
