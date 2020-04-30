#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
p1 = Normal(data)

plt.hist(data, 10, density=True)
x = np.arange(0, 100)
y = [p1.pdf(i) for i in range(0, 100)]

plt.plot(x, y, c='r')
plt.show()
