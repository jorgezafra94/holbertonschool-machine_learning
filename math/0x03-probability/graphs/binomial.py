#!/usr/bin/env python3
"""
Create our Binomial function
"""


class Binomial():
    """
    class Binomial
    """
    e = 2.7182818285

    def __init__(self, data=None, n=1, p=0.5):
        """
        constructor
        """

        if data is None:
            n = int(n)
            p = float(p)
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = p
            self.n = n

        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data)/len(data)
            variance = 0
            for elem in data:
                variance += (elem - mean) ** 2
            variance = variance / len(data)
            p = 1 - (variance / mean)

            self.n = int(round(mean / p))
            self.p = (mean/self.n)

    def pmf(self, k):
        """
        Probability Mass Function PMF
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0

        n_fact, k_fact, res_fact = 1, 1, 1

        for i in range(1, self.n + 1):
            n_fact *= i
        for i in range(1, k + 1):
            k_fact *= i
        for i in range(1, (self.n - k) + 1):
            res_fact *= i

        first = (n_fact)/(k_fact * res_fact)
        second = self.p ** k
        third = ((1 - self.p) ** (self.n - k))

        PMF = first * second * third
        return PMF

    def cdf(self, k):
        """
        Cumulative Distribution Function CDF
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        CDF = 0
        for i in range(0, k+1):
            CDF += self.pmf(i)
        return CDF
