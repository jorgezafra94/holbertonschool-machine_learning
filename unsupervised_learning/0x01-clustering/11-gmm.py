#!/usr/bin/env python3
"""
kmeans using scikit-learn
"""

import sklearn.mixture


def gmm(X, k):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
    * k is the number of clusters

    Returns: pi, m, S, clss, bic
    * pi is a numpy.ndarray of shape (k,) containing the cluster priors
    * m is a numpy.ndarray of shape (k, d) containing the centroid means
    * S is a numpy.ndarray of shape (k, d, d) containing the covariance
      matrices
    * clss is a numpy.ndarray of shape (n,) containing the cluster indices
      for each data point
    * bic is a numpy.ndarray of shape (kmax - kmin + 1) containing the BIC
      value for each cluster size tested
    """
    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    GMM.fit(X)

    m = GMM.means_
    S = GMM.covariances_
    pi = GMM.weights_
    clss = GMM.predict(X)
    BIC = GMM.bic(X)

    return pi, m, S, clss, BIC
