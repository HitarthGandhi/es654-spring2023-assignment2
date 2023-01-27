import numpy as np

from tree.base import DecisionTree

np.random.seed(1234)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps

# for plotting
# import matplotlib.pyplot as plt
# plt.plot(x, y, 'o')
# plt.plot(x, x**2 + 1, 'r-')
# plt.show()
