#!/usr/bin/env python3
"""
Moving Average using Bias correction
"""


def moving_average(data, beta):
    """
    data is the list of data to calculate the moving average of
    beta is the weight used for the moving average
    Your moving average calculation should use bias correction
    Returns: a list containing the moving averages of data
    """
    new_list = []
    V = 0
    for i in range(1, len(data) + 1):
        Vt = (beta * V) + ((1 - beta) * data[i - 1])
        correction = Vt / (1 - (beta ** i))
        new_list.append(correction)
        V = Vt
    return new_list
