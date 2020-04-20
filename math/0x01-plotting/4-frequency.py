#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins = np.arange(0, 110, 10)
# Titles
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
# config of x axis
plt.xlim(0, 100)
plt.xticks(np.arange(0, 110, 10))
# config of y axis
plt.ylim(0, 30)
# data to graphic
plt.hist(student_grades, bins=bins, edgecolor='black')
# print graphic
plt.show()
