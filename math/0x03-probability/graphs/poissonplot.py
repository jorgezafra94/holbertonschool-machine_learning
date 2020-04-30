#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
Poisson = __import__('poisson').Poisson

np.random.seed(0)
data = np.random.poisson(5., 10000).tolist()
p1 = Poisson(data)

# print(data)
fig = plt.figure()

ax1 = plt.subplot2grid((3, 1), (0, 0))
x = np.arange(0, 14)
ax1.hist(data, x, density=True)
ax1.set_title("Poisson distribution")

y = [p1.pmf(i) for i in range(0, len(x))]
print(sum(y))
z = [p1.cdf(i) for i in range(0, len(x))]
ax2 = plt.subplot2grid((3, 1), (1, 0))
ax2.plot(x, y, color='red', linestyle='-')
ax2.set_title("PMF")

ax3 = plt.subplot2grid((3, 1), (2, 0))
ax3.plot(x, z, color='green', linestyle='-')
ax3.set_title("CDF")

fig.subplots_adjust(hspace = 1)
plt.show()
