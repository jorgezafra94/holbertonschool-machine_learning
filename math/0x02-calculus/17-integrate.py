#!/usr/bin/env python3
"""
Intregal of polynomial
"""


def poly_integral(poly, C=0):
    """
    poly_integral: return the integral of a polynomial
    @poly: polynomial to integrate
    @C: constant
    """
    if (type(poly) is not list or len(poly) == 0):
        return None

    if(type(C) is not int):
        return None

    for elem in poly:
        if(type(elem) is not float and type(elem) is not int):
            return None

    integral = [C]

    for i in range(0, len(poly)):
        aux = poly[i]/(i + 1)
        if (aux).is_integer():
            aux = int(aux)
        integral.append(aux)

    return integral
