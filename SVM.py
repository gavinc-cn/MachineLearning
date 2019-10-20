import numpy as np
import matplotlib.pyplot as plt
import utils

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()

x = iris.data
y = iris.target

x = x[y < 2, :2]
y = y[y < 2]

plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue')
plt.show()

standardScaler = StandardScaler()
standardScaler.fit(x)
x_standard = standardScaler.transform(x)

svc = LinearSVC(C=1e9)
svc.fit(x_standard, y)

utils.plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(x_standard[y == 0, 0], x_standard[y == 0, 1])
plt.scatter(x_standard[y == 1, 0], x_standard[y == 1, 1])
plt.show()


























