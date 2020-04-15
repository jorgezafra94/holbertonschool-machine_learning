#!/usr/bin/env python3
"""
addition of two vectors
"""


def add_arrays(arr1, arr2):
    """
    if both arrays have the same size you can add them
    @arr1: first array
    @arr2: second array
    Return: - new array: this new array is the result of the addition
            - None: if the addition cant be realized
    """
    if (len(arr1) == len(arr2)):
        result = []
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
    else:
        return None
