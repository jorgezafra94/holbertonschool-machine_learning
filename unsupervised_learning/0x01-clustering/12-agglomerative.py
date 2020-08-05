#!/usr/bin/env python3
"""
agglomerative clustering scipy
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    * X is a numpy.ndarray of shape (n, d) containing the dataset
    * dist is the maximum cophenetic distance for all clusters
    * Performs agglomerative clustering with Ward linkage
    * Displays the dendrogram with each cluster displayed in a
      different color
    Returns: clss, a numpy.ndarray of shape (n,) containing the
    cluster indices for each data point
    """
    hierarchy = scipy.cluster.hierarchy
    linkage_mat = hierarchy.linkage(y=X, method='ward')
    fcluster = hierarchy.fcluster(Z=linkage_mat, t=dist, criterion='distance')
    plt.figure()
    hierarchy.dendrogram(linkage_mat, color_threshold=dist)
    plt.show()
    return fcluster
