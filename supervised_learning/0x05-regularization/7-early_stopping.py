#!/usr/bin/env python3
"""
Eraly stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Early stopping should occur when the validation cost of the network has not
          decreased relative to the optimal validation cost by more than the
          threshold over a specific patience count
    cost is the current validation cost of the neural network
    opt_cost is the lowest recorded validation cost of the neural network
    threshold is the threshold used for early stopping
    patience is the patience count used for early stopping
    count is the count of how long the threshold has not been met
    Returns: a boolean of whether the network should be stopped early,
             followed by the updated count
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count = count + 1
    if count == patience:
        return (True, count)
    else:
        return(False, count)
