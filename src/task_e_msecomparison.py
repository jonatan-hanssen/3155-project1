from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold

# Our own library of functions
from utils import *

np.random.seed(42069)
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)
scaling = False

# Highest order polynomial we fit with
N = 15
bootstraps = 100
K = 10
kfolds = KFold(n_splits=K)

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)
z = z.ravel()

lambdas = np.logspace(-12, -4, 6)
for i in range(len(lambdas)):
    plt.subplot(321 + i)
    plt.suptitle(f"MSE by polynomial degree for different values of lambda")

    errors_cv = np.zeros(N)
    errors_cv_scikit = np.zeros(N)
    errors_boot = np.zeros(N)

    Ridge_model = Ridge(lambdas[i], fit_intercept=scaling)
    print("Cross")
    for n in range(N):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        errors_cv[n] = crossval(X[:, :l], z, K, scaling=scaling)

        error_scikit = cross_validate(
            estimator=Ridge_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_cv_scikit[n] = np.mean(-error_scikit["test_score"])

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
            model=ridge,
            lam=lambdas[i],
        )

        error, _, _ = bias_variance(z_test, z_preds)
        errors_boot[n] = error

    plt.plot(errors_boot, label="bootstrap")
    plt.plot(errors_cv, label="cross validation implementation")
    plt.plot(errors_cv_scikit, label="cross validation skicit learn")
    plt.xlabel("Model Polynomial Degree")

    # plt.ylim(0, 0.1)
    plt.legend()

    plt.title(f"lambda = {lambdas[i]:.5}")

    plt.legend()

# Calculate scores OLS without resampling

# ---------------- PLOTTING GRAPHS --------------

# plt.subplot(222)
#
# plt.subplot(223)
# plt.plot(R2_train, label="train")
# plt.plot(R2_test, label="test")
# plt.xlabel("Polynomial degree")
# plt.legend()
# plt.title("R2 scores")


plt.show()

# linear_regression(x,y,z)
