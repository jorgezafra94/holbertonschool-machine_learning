#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
Exponential = __import__('exponential').Exponential

np.random.seed(0)
s = np.random.exponential(0.4, 1000).tolist()
p1 = Exponential(s)

fig = plt.figure()

ax1 = plt.subplot2grid((3, 1), (0, 0))
x = np.linspace(0, 1.5, 20)
ax1.hist(s, x, density=True)
ax1.set_title("Exponential distribution")

y = [p1.pdf(i) for i in range(0, len(x))]
print(sum(y))
z = [p1.cdf(i) for i in range(0, len(x))]
ax2 = plt.subplot2grid((3, 1), (1, 0))
ax2.plot(x, y, color='red', linestyle='-')
ax2.set_ylim(0, 1)
ax2.set_xlim(0, 0.5)
ax2.set_title("PDF")

ax3 = plt.subplot2grid((3, 1), (2, 0))
ax3.plot(x, z, color='green', linestyle='-')
ax3.set_title("CDF")

fig.subplots_adjust(hspace = 1)
plt.show()
