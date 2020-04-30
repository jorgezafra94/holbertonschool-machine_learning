#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
Normal = __import__('normal').Normal


np.random.seed(0)
data = np.random.normal(30, 5, 1000).tolist()
n = Normal(data)

x = np.arange(0, 50, 0.001)
y = [n.pdf(x) for x in x]
z = [n.cdf(x) for x in x]

fig = plt.figure()
ax1 = plt.subplot2grid((3,1), (0,0))
ax1.hist(data, 25, density=True)
ax2 = plt.subplot2grid((3,1), (1,0))
ax2.plot(x, y, c='r')
ax2.set_xticks(np.arange(5, 50, 5))
ax3 = plt.subplot2grid((3,1), (2,0))
ax3.plot(x, z, c='g')
ax3.set_ylim(-5, 5)
plt.show()
