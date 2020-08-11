#!/usr/bin/env python3
"""
backward Algorithm
Read:
1) http://www.adeveloperdiary.com/data-science/machine-learning->
->forward-and-backward-algorithm-in-hidden-markov-model/

2) https://web.stanford.edu/~jurafsky/slp3/A.pdf
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    * Observation is a numpy.ndarray of shape (T,) that contains the index
      of the observation
      - T is the number of observations
    * Emission is a numpy.ndarray of shape (N, M) containing the emission
      probability of a specific observation given a hidden state
      - Emission[i, j] is the probability of observing j given the hidden
        state i
      - N is the number of hidden states
      - M is the number of all possible observations
    * Transition is a 2D numpy.ndarray of shape (N, N) containing the
      transition probabilities
      - Transition[i, j] is the probability of transitioning from the
        hidden state i to j
    * Initial a numpy.ndarray of shape (N, 1) containing the probability
      of starting in a particular hidden state

    Returns: P, B, or None, None on failure
    * P is the likelihood of the observations given the model
    * B is a numpy.ndarray of shape (N, T) containing the backward path
      probabilities
      - B[i, j] is the probability of generating the future observations
        from hidden state i at time j
    """
    # ****************** observation conditionals ************************
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    for elem in Observation:
        if elem < 0:
            return None, None
    # ******************* emission conditionals *******************
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    sum1 = np.sum(Emission, axis=1)
    if not (sum1 == 1.).all():
        return None, None
    # ******************* transition conditionals *******************
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    sum2 = np.sum(Transition, axis=1)
    if not (sum2 == 1.).all():
        return None, None
    # ******************* initial conditionals *******************
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    sum3 = np.sum(Initial)
    if sum3 != 1.:
        return None, None
    # ******************* shape conditionals *******************
    if Initial.shape[0] != Transition.shape[0]:
        return None, None

    if Emission.shape[0] != Transition.shape[0]:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]

    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for n in range(N):
            trans = Transition[n]
            em = Emission[:, Observation[t + 1]]
            post = beta[:, t + 1]
            first = post * em
            beta[n, t] = np.dot(first.T, trans)

    prob = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return (prob, beta)
