#!/usr/bin/env python3
"""
based on:
* https://medium.com/samkirkiles/reinforce-policy-gradients-
  from-scratch-in-numpy-6a09ae0dfe12

Policy Gradient
"""

import numpy as np


def policy(matrix, weight):
    """
    * matrix: states
    * weights: weights of the states
    Return: the new policy
    """

    result = np.dot(matrix, weight)
    result = np.exp(result)
    my_policy = result / result.sum()
    return my_policy


def policy_gradient(state, weight):
    """
    Monte-Carlo policy gradient
    * state: matrix representing the current observation of the environment
    * weight: matrix of random weight
    Return: the action and the gradient
    """

    def softmax_grad(softmax):
        s = softmax.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

    probs = policy(state, weight)
    action = np.argmax(probs)
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[0, action]
    grad = state.T.dot(dlog[None, :])

    return (action, grad)
