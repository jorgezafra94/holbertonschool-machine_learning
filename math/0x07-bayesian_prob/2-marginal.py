#!/usr/bin/env python3
"""
marginal prob
"""

import numpy as np


def marginal(x, n, P, Pr):
    """
    * x is the number of patients that develop severe side effects
    * n is the total number of patients observed
    * P is a 1D numpy.ndarray containing the various hypothetical
      probabilities of patients developing severe side effects
    * Pr is a 1D numpy.ndarray containing the prior beliefs about P
    * If n is not a positive integer, raise a ValueError with the
      message n must be a positive integer
    * If x is not an integer that is greater than or equal to 0,
      raise a ValueError with the message x must be an integer that
      is greater than or equal to 0
    * If x is greater than n, raise a ValueError with the message x
      cannot be greater than n
    * If P is not a 1D numpy.ndarray, raise a TypeError with the
      message P must be a 1D numpy.ndarray
    * If Pr is not a numpy.ndarray with the same shape as P, raise a
      TypeError with the message Pr must be a numpy.ndarray with the
      same shape as P
    * If any value in P or Pr is not in the range [0, 1], raise a
      ValueError with the message All values in {P} must be in the
      range [0, 1] where {P} is the incorrect variable
    * If Pr does not sum to 1, raise a ValueError with the message Pr
      must sum to 1
    * All exceptions should be raised in the above order
    Returns: the marginal probability of obtaining x and n
    """
    if type(n) is not int or n <= 0:
        raise ValueError('n must be a positive integer')

    if type(x) is not int or x < 0:
        m = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(m)

    if x > n:
        raise ValueError('x cannot be greater than n')

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')

    if not isinstance(Pr, np.ndarray) or P.shape != Pr.shape:
        m = 'Pr must be a numpy.ndarray with the same shape as P'
        raise TypeError(m)

    for i in range(len(P)):
        if not (P[i] >= 0 and P[i] <= 1):
            a = 'All values in P must be in the range [0, 1]'
            raise ValueError(a)

        if not (Pr[i] >= 0 and Pr[i] <= 1):
            a = 'All values in Pr must be in the range [0, 1]'
            raise ValueError(a)

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')

    # likelihood
    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)

    combination = fact_n / (fact_x * fact_nx)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    # Prior
    prior = Pr

    # Intersection
    intersection = prior * likelihood

    # Marginal
    marginal = np.sum(intersection)

    return marginal
