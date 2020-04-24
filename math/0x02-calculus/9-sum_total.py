#!/usr/bin/env python3
"""
using seegma logic in order to add each result
"""


def recursion(n, actual, value):
    """
    recursion: use seegma locig with function i^2
    @n: top of seegma
    @actual: actual value in seegma function
    @value: actual result
    Return: result of seegma function
    """
    if (n < actual):
        return value
    else:
        value += (actual ** 2)
        result = recursion(n, actual+1, value)
    return result


def summation_i_squared(n):
    """
    summation_i_squared: is a function that uses i^2 function in seegma logic
                        the seegma will start in i=1
    @n: will be the top of seegma function
    Return: result of function seegma
    """
    if not n:
        return None
    if (type(n) is not):
        return None
    result = recursion(n, 1, 0)
    return result
