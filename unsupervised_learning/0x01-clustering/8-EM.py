#!/usr/bin/env python3
"""
estimation and maximization step of GMM
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * k is a positive integer containing the number of clusters
    * iterations is a positive integer containing the maximum number of
      iterations for the algorithm
    * tol is a non-negative float containing tolerance of the log
      likelihood, used to determine early stopping i.e. if the difference
      is less than or equal to tol you should stop the algorithm
    * verbose is a boolean that determines if you should print information
      about the algorithm
    * If True, print Log Likelihood after {i} iterations: {l} every 10
      iterations and after the last iteration
      - {i} is the number of iterations of the EM algorithm
      - {l} is the log likelihood
    * You may use at most 1 loops

    Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    * pi is a numpy.ndarray of shape (k,) containing the priors for each
      cluster
    * m is a numpy.ndarray of shape (k, d) containing the centroid means
      for each cluster
    * S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices for each cluster
    * g is a numpy.ndarray of shape (k, n) containing the probabilities
      for each data point in each cluster
    * l is the log likelihood of the model
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None, None)

    if type(k) is not int or k <= 0:
        return (None, None, None, None, None)

    if type(iterations)is not int or iterations <= 0:
        return (None, None, None, None, None)

    if type(tol) is not float or tol < 0:
        return (None, None, None, None, None)

    if type(verbose) is not bool:
        return (None, None, None, None, None)

    pi, m, S = initialize(X, k)
    l_init = 0
    count = 0
    g, log_like = expectation(X, pi, m, S)
    while(count < iterations):
        if (np.abs(l_init - log_like)) <= tol:
            break

        if verbose is True and count % 10 == 0:
            m1 = 'Log Likelihood after {}'.format(count)
            m2 = ' iterations: {}'.format(log_like.round(5))
            print(m1 + m2)

        l_init = log_like
        pi, m, S = maximization(X, g)
        g, log_like = expectation(X, pi, m, S)

        count += 1

    if verbose is True:
        m1 = 'Log Likelihood after {}'.format(count)
        m2 = ' iterations: {}'.format(log_like.round(5))
        print(m1 + m2)

    return (pi, m, S, g, log_like)
