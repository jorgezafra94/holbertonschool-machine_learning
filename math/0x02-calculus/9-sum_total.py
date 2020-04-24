#!/usr/bin/env python3
"""
using seegma logic in order to add each result
"""


def summation_i_squared(n):
    """
    summation_i_squared: is a function that uses i^2 function in seegma logic
                        the seegma will start in i=1
    @n: will be the top of seegma function
    Return: result of function seegma
    """
    if n is None or n < 0:
        return None
    result = list(map(lambda x: x**2, list(range(1, n+1))))
    result = sum(result)
    return result
