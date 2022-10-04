from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

# Our own library of functions
from utils import *

np.random.seed(42069)
# Make data.
x = np.arange(0, 1, 0.1)
y = np.arange(0, 1, 0.1)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)

# Highest order polynomial we fit with
N = 6
bootstraps = 100

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

num_lambdas = 20
lambdas = np.logspace(-10, 0, num_lambdas)
errors = np.zeros(num_lambdas)
print(np.format_float_scientific(lambdas[1], 3))
print(lambdas[1])
xi = [i for i in range(0, num_lambdas, 2)]
# print(lambdas[::2])
for i in range(num_lambdas):
    print(i)

    z_preds = bootstrap(
        X,
        X_train,
        X_test,
        z_train,
        z_test,
        bootstraps,
        scaling=True,
        model=ridge,
        lam=lambdas[i],
    )

    error, _, _ = bias_variance(z_test, z_preds)
    errors[i] = error

plt.title("MSE at polynomial deg. 6 for different values of lambda")
plt.plot(errors, label="MSE test")
plt.xlabel("Value of lambda")
plt.ylabel("MSE")
plt.xticks(
    xi, [float(np.format_float_scientific(elem, precision=3)) for elem in lambdas[::2]]
)

plt.legend()

plt.show()

# linear_regression(x,y,z)
