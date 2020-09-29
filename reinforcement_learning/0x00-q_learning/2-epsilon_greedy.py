#!/usr/bin/env python3
"""
based on:
https://www.learndatasci.com/tutorials/
reinforcement-q-learning-scratch-python-openai-gym/
after the Q-table declaration inside the while loop

Training Agent, using epsilon greedy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    * Q is a numpy.ndarray containing the q-table
    * state is the current state
    * epsilon is the epsilon to use for the calculation
    * You should sample p with numpy.random.uniformn to
      determine if your algorithm should explore or exploit
    * If exploring, you should pick the next action with
      numpy.random.randint from all possible actions

    Returns: the next action index
    """
    if np.random.uniform(0, 1) < epsilon:
        # Explore action space randomly
        # the shape of Q-table is (num of states, num of actions)
        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit learned values by choosing optimal values
        # as we know states are rows columns actions
        action = np.argmax(Q[state])

    return action
