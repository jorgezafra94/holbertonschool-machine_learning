#!/usr/bin/env python3
"""
 Baum-Welch
 read:
1) http://www.adeveloperdiary.com/data-science/machine-learning/->
->implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    forward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]
    alpha = np.zeros((N, T))
    aux = (Initial.T * Emission[:, Observation[0]])
    alpha[:, 0] = aux.reshape(-1)
    for t in range(1, T):
        for n in range(N):
            prev = alpha[:, t - 1]
            trans = Transition[:, n]
            em = Emission[n, Observation[t]]
            alpha[n, t] = np.sum(prev * trans * em)

    prop = np.sum(alpha[:, -1])
    return (prop, alpha)


def backward(Observation, Emission, Transition, Initial):
    """
    backward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]

    # we should initialize the last prob as 1
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    # this should be start in T-2 and should stop in 0
    # that is why the range has this form and step
    for t in range(T - 2, -1, -1):
        for n in range(N):
            trans = Transition[n]
            em = Emission[:, Observation[t + 1]]
            post = beta[:, t + 1]
            first = post * em
            beta[n, t] = np.dot(first.T, trans)

    prob = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return (prob, beta)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    * Observations is a numpy.ndarray of shape (T,) that contains the index of
      the observation
      - T is the number of observations
    * Transition is a numpy.ndarray of shape (M, M) that contains the
      initialized
      transition probabilities
      - M is the number of hidden states
    * Emission is a numpy.ndarray of shape (M, N) that contains the initialized
      emission probabilities
      - N is the number of output states
    * Initial is a numpy.ndarray of shape (M, 1) that contains the initialized
      starting probabilities
    * iterations is the number of times expectation-maximization should be
      performed

    Returns: the converged Transition, Emission, or None, None on failure
    """
    # ****************** observation conditionals ************************
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    for elem in Observations:
        if elem < 0:
            return None, None
    # ******************* emission conditionals *******************
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None

    # ******************* transition conditionals *******************
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    # ******************* initial conditionals *******************
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    # ******************* iterations **************************
    if type(iterations) is not int or iterations <= 0:
        return None, None
    # ******************* shape conditionals *******************
    if Initial.shape[0] != Transition.shape[0]:
        return None, None

    if Emission.shape[0] != Transition.shape[0]:
        return None, None

    N, M = Emission.shape
    T = Observations.shape[0]

    Obs = Observations.copy()
    Init = Initial.copy()
    Emi = Emission.copy()
    Trans = Transition.copy()

    for n in range(iterations):

        _, alpha = forward(Obs, Emi, Trans, Init)
        _, beta = backward(Obs, Emi, Trans, Init)

        xi = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            first = np.dot(alpha[:, t].T, Trans)
            second = Emi[:, Obs[t + 1]].T
            denominator = np.dot(first * second, beta[:, t + 1])

            for i in range(N):
                first = alpha[i, t] * Trans[i]
                second = Emi[:, Obs[t + 1]].T
                numerator = first * second * beta[:, t + 1].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        Trans = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2],
                                         axis=0).reshape((-1, 1))))
        denominator = np.sum(gamma, axis=1)
        for l in range(M):
            Emi[:, l] = np.sum(gamma[:, Obs == l], axis=1)
        Emi = np.divide(Emi, denominator.reshape((-1, 1)))

    return (Trans, Emi)
