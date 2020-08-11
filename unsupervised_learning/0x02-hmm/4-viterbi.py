#!/usr/bin/env python3
"""
Viretbi Algorithm
Read:
1) http://www.adeveloperdiary.com/data-science/machine-learning/->
implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/

2) https://web.stanford.edu/~jurafsky/slp3/A.pdf
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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

    Returns: path, P, or None, None on failure
    * path is the a list of length T containing the most likely sequence
      of hidden states
    * P is the probability of obtaining the path sequence
    """
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

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    aux = (Initial.T * Emission[:, Observation[0]])
    viterbi[:, 0] = aux.reshape(-1)

    backpointer[:, 0] = 0

    for t in range(1, T):
        for n in range(N):
            prev = viterbi[:, t - 1]
            trans = Transition[:, n]
            em = Emission[n, Observation[t]]
            result = prev * trans * em
            viterbi[n, t] = np.amax(result)
            backpointer[n, t - 1] = np.argmax(result)

    path = []
    # Find the most probable last hidden state
    last_state = np.argmax(viterbi[:, T - 1])
    path.append(int(last_state))

    # backtracking algorithm gotten from first read
    for i in range(T - 2, -1, -1):
        path.append(int(backpointer[int(last_state), i]))
        last_state = backpointer[int(last_state), i]

    # Flip the path array since we were backtracking
    path.reverse()

    min_prob = np.amax(viterbi, axis=0)
    min_prob = np.amin(min_prob)

    return (path, min_prob)
