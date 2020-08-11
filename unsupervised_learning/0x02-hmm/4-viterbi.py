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

    N, M = Emission.shape
    T = Observation.shape[0]

    viterbi = np.zeros((N, T))
    backpointer = np.zeros((N, T))

    aux = (Initial * Emission[:, Observation[0]].reshape(-1, 1))
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
