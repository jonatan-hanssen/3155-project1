from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# Our own library of functions
from utils import *

np.random.seed(42069)
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)

# Highest order polynomial we fit with
N = 15
bootstraps = 100

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

lambdas = np.logspace(-12, -4, 6)
for i in range(len(lambdas)):
    plt.subplot(321 + i)
    plt.suptitle(f"MSE by polynomial degree for different values of lambda")
    model_Lasso = Lasso(lambdas[i], max_iter=200)

    errors = np.zeros(N)
    biases = np.zeros(N)
    variances = np.zeros(N)

    for n in range(N):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        z_preds = bootstrap(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            z_test,
            bootstraps,
            scaling=True,
            model=model_Lasso,
            lam=lambdas[i],
        )

        error, bias, variance = bias_variance(z_test, z_preds)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    print(f"{variances=}")
    print(f"{biases=}")
    print(f"{errors=}")
    plt.plot(errors, label="MSE test")
    plt.plot(biases, label="bias")
    plt.plot(variances, label="variance")
    plt.ylim(0, 0.1)
    plt.xlabel("Polynomial Degree")
    plt.tight_layout(h_pad=0.001)

    plt.title(f"lambda = {lambdas[i]:.5}")

    plt.legend()

plt.show()

# linear_regression(x,y,z)
