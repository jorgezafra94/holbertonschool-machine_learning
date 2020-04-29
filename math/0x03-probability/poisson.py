#!/usr/bin/env python3
"""
Create our Poisson function
"""


class Poisson():
    """
    class Poisson
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        constructor
        """

        if data is None:
            lambtha = float(lambtha)
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """
        Probability Mass Function PMF
        """
        if type(k) is int or type(k) is float:
            k = int(k)
            if k < 0:
                return 0
            den = 1
            for i in range(1, k+1):
                den = den * i
            PMF = ((Poisson.e ** (-self.lambtha)) * (self.lambtha ** k))/den
            return PMF

    def cdf(self, k):
        """
        Cumulative Distribution Function CDF
        """
        if type(k) is int or type(k) is float:
            k = int(k)
            if k < 0:
                return 0
            CDF = 0
            for i in range(0, k+1):
                CDF += self.pmf(i)
            return CDF
