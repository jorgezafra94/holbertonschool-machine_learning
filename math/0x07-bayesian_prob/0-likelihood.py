#!/usr/bin/env python3
"""
Likelihood
"""

import numpy as np


def likelihood(x, n, P):
    """
    * x is the number of patients that develop severe
      side effects
    * n is the total number of patients observed
    * P is a 1D numpy.ndarray containing the various
      hypothetical probabilities of developing severe
      side effects
    * If n is not a positive integer, raise a ValueError
      with the message n must be a positive integer
    * If x is not an integer that is greater than or equal
      to 0, raise a ValueError with the message x must be
      an integer that is greater than or equal to 0
    * If x is greater than n, raise a ValueError with the
      message x cannot be greater than n
    * If P is not a 1D numpy.ndarray, raise a TypeError
      with the message P must be a 1D numpy.ndarray
    * If any value in P is not in the range [0, 1], raise
      a ValueError with the message All values in P must
      be in the range [0, 1]
    Returns: a 1D numpy.ndarray containing the likelihood
    of obtaining the data, x and n, for each probability
    in P, respectively
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

    for elem in P:
        if not (elem >= 0 and elem <= 1):
            a = 'All values in P must be in the range [0, 1]'
            raise ValueError(a)

    # as x follows a Binomial distribution we have to use
    # the equation of binomial distribution
    # f = (n! / (x! * (n-x)!)) * (p ** x) * (1-p) ** (n-x)
    # where x is the point to evaluate
    # p is the probability
    # n is the total of samples

    fact_n = np.math.factorial(n)
    fact_x = np.math.factorial(x)
    fact_nx = np.math.factorial(n - x)

    combination = fact_n / (fact_x * fact_nx)
    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
