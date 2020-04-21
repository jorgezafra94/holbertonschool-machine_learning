#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

name = ['apples', 'bananas', 'oranges', 'peaches']
color = ['red', 'yellow', '#ff8000', '#ffe5b4']
x = np.arange(0, 3)
new = np.array([0, 0, 0])

for i in range(len(fruit)):
    plt.bar(x, fruit[i], width=0.5, color=color[i], bottom=new, label=name[i])
    new = np.add(new, fruit[i])

plt.xticks(x, ['Farrah', 'Fred', 'Felicia'])
plt.ylabel("Quantity of Fruit")
plt.yticks(np.arange(0, 90, 10))
plt.title("Number of Fruit per Person")
plt.legend()
plt.show()
