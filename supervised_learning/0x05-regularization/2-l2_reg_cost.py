#!/usr/bin/env python3
"""
L2 cost in Tensorflow
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    * cost is a tensor containing the cost of the network without
         L2 regularization
    * Returns: a tensor containing the cost of the network accounting
         for L2 regularization
    """
    cost_L2 = tf.losses.get_regularization_losses(scope=None)
    return (cost + cost_L2)
