#!/usr/bin/env python3
"""
Forward Markov Chain
Read:
http://www.adeveloperdiary.com/data-science/machine-learning->
->forward-and-backward-algorithm-in-hidden-markov-model/
and
https://web.stanford.edu/~jurafsky/slp3/A.pdf
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    * Observation is a numpy.ndarray of shape (T,) that contains the index
      of the observation
      - T is the number of observations
    * Emission is a numpy.ndarray of shape (N, M) containing the emission
      probability of a specific observation given a hidden state
      - Emission[i, j] is the probability of observing j given the
        hidden state i
      - N is the number of hidden states
      - M is the number of all possible observations
    * Transition is a 2D numpy.ndarray of shape (N, N) containing the
      transition probabilities
      - Transition[i, j] is the probability of transitioning from
        the hidden state i to j
    * Initial a numpy.ndarray of shape (N, 1) containing the probability
      of starting in a particular hidden state

    Returns: P, F, or None, None on failure
    * P is the likelihood of the observations given the model
    * F is a numpy.ndarray of shape (N, T) containing the forward path
      probabilities
      - F[i, j] is the probability of being in hidden state i at time j
        given the previous observations
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
