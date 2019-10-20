import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import utils

x, y = datasets.make_moons()

print(x.shape)
print(y.shape)

plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

x, y = datasets.make_moons(noise=0.15, random_state=666)
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()

# 使用多项式特征的SVM


def polynomial_svc(degree, C=1.0):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('std_scaler', StandardScaler()),
        ('linearSVC', LinearSVC(C=C))
    ])


poly_svc = polynomial_svc(degree=3)
poly_svc.fit(x, y)

utils.plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()


# 使用多项式核函数SVM

def polynomial_kernel_svc(degree, C=1.0):
    return Pipeline([
        ('std_scaler', StandardScaler()),
        ('kernelSVC', SVC(kernel='poly', degree=degree, C=C))
    ])


poly_kernel_svc = polynomial_kernel_svc(degree=3)
poly_kernel_svc.fit(x, y)


utils.plot_decision_boundary(poly_kernel_svc, axis=[-1.5, 2.5, -1, 1.5])
plt.scatter(x[y==0, 0], x[y==0, 1])
plt.scatter(x[y==1, 0], x[y==1, 1])
plt.show()








































