#!/usr/bin/env python3
"""
Complete Model Optimization
"""

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    the shuffled X and Y matrices
    """
    vector = np.random.permutation(np.arange(X.shape[0]))
    X_shu, Y_shu = X, Y
    X_shu = X[vector]
    Y_shu = Y[vector]
    return X_shu, Y_shu


def create_batch_norm_layer(prev, n, activation):
    """
    batch_nomalization
    """
    w_init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=w_init)
    Z = layers(prev)
    gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[n]),
                        name='gamma', trainable=True)
    beta = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n]),
                       name='beta', trainable=True)
    epsilon = tf.constant(1e-8)
    mean, variance = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(x=Z, mean=mean, variance=variance,
                                       offset=beta, scale=gamma,
                                       variance_epsilon=epsilon)
    if not activation:
        return Z_norm
    else:
        A = activation(Z_norm)
        return A


def forward_prop(x, layers, activations):
    """
    forward propagation
    """
    A = create_batch_norm_layer(x, layers[0], activations[0])
    for i in range(1, len(activations)):
        A = create_batch_norm_layer(A, layers[i], activations[i])
    return A


def calculate_accuracy(y, y_pred):
    """
    accuracy of the prediction
    """
    index_y = tf.math.argmax(y, axis=1)
    index_pred = tf.math.argmax(y_pred, axis=1)
    comp = tf.math.equal(index_y, index_pred)
    cast = tf.cast(comp, dtype=tf.float32)
    accuracy = tf.math.reduce_mean(cast)
    return accuracy


def calculate_loss(y, y_pred):
    """
    loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step):
    """
    Adam optimization
    """
    a = tf.train.AdamOptimizer(learning_rate=alpha, beta1=beta1,
                               beta2=beta2, epsilon=epsilon)
    optimize = a.minimize(loss, global_step=global_step)
    return optimize


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    learning rate decay
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    Data_train is a tuple containing the training inputs and
               training labels, respectively
    Data_valid is a tuple containing the validation inputs and
               validation labels, respectively
    layers is a list containing the number of nodes in each
               layer of the network
    activation is a list containing the activation functions
               used for each layer of the network
    alpha is the learning rate
    beta1 is the weight for the first moment of Adam Optimization
    beta2 is the weight for the second moment of Adam Optimization
    epsilon is a small number used to avoid division by zero
    decay_rate is the decay rate for inverse time decay of
               the learning rate (the corresponding decay step should be 1)
    batch_size is the number of data points that should be in a mini-batch
    epochs is the number of times the training should pass
               through the whole dataset
    save_path is the path where the model should be saved to
    Returns: the path where the model was saved

    """
    # building model
    x = tf.placeholder(tf.float32, shape=[None, Data_train[0].shape[1]],
                       name='x')
    tf.add_to_collection('x', x)
    y = tf.placeholder(tf.float32, shape=[None, Data_train[1].shape[1]],
                       name='y')
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # getting data_batch
    mini_iter = Data_train[0].shape[0] / batch_size
    if (mini_iter).is_integer():
        mini_iter = int(mini_iter)
    else:
        mini_iter = int(mini_iter) + 1

    # Adam training & learning decay
    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, mini_iter)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step)
    tf.add_to_collection('train_op', train_op)

    # global initialization
    train_feed = {x: Data_train[0], y: Data_train[1]}
    valid_feed = {x: Data_valid[0], y: Data_valid[1]}
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as ses:
        ses.run(init)
        for i in range(epochs + 1):
            T_cost = ses.run(loss, train_feed)
            T_acc = ses.run(accuracy, train_feed)
            V_cost = ses.run(loss, valid_feed)
            V_acc = ses.run(accuracy, valid_feed)
            print("After {} epochs:".format(i))
            print('\tTraining Cost: {}'.format(T_cost))
            print('\tTraining Accuracy: {}'.format(T_acc))
            print('\tValidation Cost: {}'.format(V_cost))
            print('\tValidation Accuracy: {}'.format(V_acc))

            if i < epochs:
                X_shu, Y_shu = shuffle_data(Data_train[0], Data_train[1])
                for j in range(mini_iter):
                    ini = j * batch_size
                    fin = (j + 1) * batch_size
                    if fin > Data_train[0].shape[0]:
                        fin = Data_train[0].shape[0]
                    mini_feed = {x: X_shu[ini:fin], y: Y_shu[ini:fin]}

                    ses.run(train_op, mini_feed)
                    if j != 0 and (j + 1) % 100 == 0:
                        Min_cost = ses.run(loss, mini_feed)
                        Min_acc = ses.run(accuracy, mini_feed)
                        print("\tStep {}:".format(j + 1))
                        print("\t\tCost: {}".format(Min_cost))
                        print("\t\tAccuracy: {}".format(Min_acc))
                ses.run(alpha)
        save_path = saver.save(ses, save_path)
    return save_path