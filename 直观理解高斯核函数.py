import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-4, 5, 1)
print(x)

y = np.array((x >= -2) & (x <= 2), dtype='int')
print(y)

plt.scatter(x[y==0], [0]*len(x[y==0]))
plt.scatter(x[y==1], [0]*len(x[y==1]))
plt.show()


def gaussian(x, l):
    gamma = 1.0
    return np.exp(-gamma * (x-l)**2)


l1, l2 = -1, 1

x_new = np.empty((len(x), 2))
for i, data in enumerate(x):
    x_new[i, 0] = gaussian(data, l1)
    x_new[i, 1] = gaussian(data, l2)

plt.scatter(x_new[y == 0, 0], x_new[y == 0, 1])
plt.scatter(x_new[y == 1, 0], x_new[y == 1, 1])
plt.show()

