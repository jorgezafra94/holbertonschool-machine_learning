#!/usr/bin/env python3
"""
get it from:
https://datascience.stackexchange.com/questions
/51065/what-is-the-positional-encoding-in-the-transformer-model
Positional Encoding
"""

import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """
    max_seq_len is an integer representing the maximum sequence length
    dm is the model depth

    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the
    positional encoding vectors
    """
    result = np.zeros((max_seq_len, dm))

    for i in range(max_seq_len):
        for j in range(0, dm, 2):
            div_term = np.exp(j * -np.log(10000.0) / dm)
            result[i, j] = np.sin(i * div_term)
            result[i, j + 1] = np.cos(i * div_term)

    return tf.cast(result, dtype=tf.float32)
