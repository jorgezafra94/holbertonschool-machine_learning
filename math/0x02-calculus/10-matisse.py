#!/usr/bin/env python3
"""
creating a program to derivate
"""


def poly_derivative(poly):
    """
    poly_derivative: return the derivate of a polynomial
    @poly: polynomial to derivate
    """
    if(type(poly) is not list or len(poly) == 0):
        return None

    for elem in poly:
        if(type(elem) is not int and type(elem) is not float):
            return None

    if(len(poly) == 1):
        return [0]

    derivate = []
    for i in range(1, len(poly)):
        aux = poly[i] * i
        derivate.append(aux)

    return derivate
