#!/usr/bin/env python3
"""
define Positive, Negative, Semidefine or indefine
"""

import numpy as np


def definiteness(matrix):
    """
    * matrix is a numpy.ndarray of shape (n, n) whose definiteness
      should be calculated
    * Return: the string Positive definite, Positive semi-definite,
        Negative semi-definite, Negative definite, or Indefinite if the
        matrix is positive definite, positive semi-definite, negative
        semi-definite, negative definite of indefinite respectively
    If matrix does not fit any of the above categories, return None
    """

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or (matrix.shape[0] != matrix.shape[1]):
        return None
    if not np.all(matrix.T == matrix):
        return None
    eigenvalues, eigenvector = np.linalg.eig(matrix)

    positive = np.where(eigenvalues > 0)
    ceros = np.where(eigenvalues == 0)
    negative = np.where(eigenvalues < 0)

    pos = eigenvalues[positive]
    cer = eigenvalues[ceros]
    neg = eigenvalues[negative]
    if pos.size and not cer.size and not neg.size:
        return ('Positive definite')
    elif pos.size and cer.size and not neg.size:
        return ('Positive semi-definite')
    elif not pos.size and not cer.size and neg.size:
        return ('Negative definite')
    elif not pos.size and cer.size and neg.size:
        return ('Negative semi-definite')
    elif pos.size and not cer.size and neg.size:
        return ('Indefinite')
    else:
        return None
