#!/usr/bin/env python3
"""
Create our Normal function
"""


class Normal():
    """
    class Normal
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0, stddev=1.):
        """
        constructor
        """

        if data is None:
            mean = float(mean)
            stddev = float(stddev)

            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = mean
            self.stddev = stddev

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data)/len(data)
            addition = 0
            for i in range(0, len(data)):
                addition += ((data[i] - self.mean) ** 2)
            self.stddev = ((addition / len(data)) ** (1/2))

    def z_score(self, x):
        """
        Z Score from x values, where Z = 0 is x = median
        """
        return ((x - self.mean)/self.stddev)

    def x_value(self, z):
        """
        x value from z score, where Z = 0 is x = median
        """
        return ((z * self.stddev) + self.mean)

    def pdf(self, x):
        """
        Probability Density Function PDF
        """
        first = 1/(self.stddev * ((2 * Normal.pi) ** (1/2)))
        second = -((x - self.mean) ** 2)/(2 * (self.stddev ** 2))
        return (first * (Normal.e ** second))

    def cdf(self, x):
        """
        Cumulative Distribution function
        """
        X = ((x - self.mean)/(self.stddev * (2 ** (1/2))))
        erf1 = (2/(Normal.pi ** (1/2)))
        erf2 = X - (X ** 3)/3 + (X ** 5)/10 - (X ** 7)/42 + (X ** 9) / 216
        erf = erf1 * erf2
        CDF = (1/2) * (1 + erf)
        return CDF
