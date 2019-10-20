import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import utils

X, y = datasets.make_moons(noise=0.15, random_state=666)

plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()


def rbf_kernel_svc(gammma=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', gamma=gammma)),
    ])


svc = rbf_kernel_svc(gammma=1)
svc.fit(X, y)

utils.plot_decision_boundary(svc, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.show()


