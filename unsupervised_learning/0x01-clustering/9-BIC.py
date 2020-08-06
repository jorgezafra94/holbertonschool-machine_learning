#!/usr/bin/env python3
"""
optimization of K using bayes in GMM
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    * X is a numpy.ndarray of shape (n, d) containing the data set
    * kmin is a positive integer containing the minimum number of clusters
      to check for (inclusive)
    * kmax is a positive integer containing the maximum number of clusters
      to check for (inclusive)
    * iterations is a positive integer containing the maximum number of
      iterations for the EM algorithm
    * tol is a non-negative float containing the tolerance for the EM
      algorithm
    * verbose is a boolean that determines if the EM algorithm should print
      information to the standard output
    * You may use at most 1 loop

    Returns: best_k, best_result, l, b, or None, None, None, None on failure
    * best_k is the best value for k based on its BIC
    * best_result is tuple containing pi, m, S
    * pi is a numpy.ndarray of shape (k,) containing the cluster priors for
      the best number of clusters
    * m is a numpy.ndarray of shape (k, d) containing the centroid means for
      the best number of clusters
    * S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices for the best number of clusters
    * l is a numpy.ndarray of shape (kmax - kmin + 1) containing the log
      likelihood for each cluster size tested
    * b is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC value
      for each cluster size tested
    * Use: BIC = p * ln(n) - 2 * l
    * p is the number of parameters required for the model
    * n is the number of data points used to create the model
    * l is the log likelihood of the model
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return (None, None, None, None)

    if type(kmin) is not int or kmin <= 0:
        return (None, None, None, None)

    if kmax is not None and type(kmax) is not int:
        return (None, None, None, None)

    if kmax is None:
        kmax = X.shape[0]

    if kmin >= kmax or kmax <= 0:
        return (None, None, None, None)

    if type(iterations) is not int or iterations <= 0:
        return (None, None, None, None)

    if type(tol) is not float or tol < 0:
        return (None, None, None, None)

    if type(verbose) is not bool:
        return (None, None, None, None)

    n, d = X.shape

    all_l = np.zeros(kmax - kmin + 1)
    all_BIC = np.zeros(kmax - kmin + 1)
    count = 0
    # lists to save all the results per k
    all_pi = []
    all_S = []
    all_m = []

    for i in range(kmin, kmax + 1):
        pi, m, S, _, log_like = expectation_maximization(X, i, iterations,
                                                         tol, verbose)
        all_pi.append(pi)
        all_m.append(m)
        all_S.append(S)
        all_l[count] = log_like
        # number of parameters in GMM
        # I got this equation here: https://www.mdpi.com/2227-7390/8/3/373/htm
        p = (i - 1) + (i * d) + ((i * d) * (d + 1) / 2)
        # getting BIC
        BIC = (p) * np.log(n) - 2 * log_like
        all_BIC[count] = BIC
        count = count + 1

    # the min BIC is the best model
    min_BIC = np.argmin(all_BIC)
    best_pi = all_pi[min_BIC]
    best_m = all_m[min_BIC]
    best_S = all_S[min_BIC]
    best_k = kmin + min_BIC

    return [best_k, (best_pi, best_m, best_S), all_l, all_BIC]
