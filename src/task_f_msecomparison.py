from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.linear_model import Ridge, Lasso, LinearRegression
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
scaling = True

# Highest order polynomial we fit with
N = 15
K = 10
kfolds = KFold(n_splits=K)

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)
z = z.ravel()

lambdas = np.logspace(-12, -4, 6)
for i in range(len(lambdas)):
    plt.subplot(321 + i)
    plt.suptitle(f"MSE by polynomial degree for OLS, Ridge and Lasso regression")

    errors_OLS = np.zeros(N)
    errors_Ridge = np.zeros(N)
    errors_Lasso = np.zeros(N)

    Ridge_model = Ridge(lambdas[i], fit_intercept=scaling)
    Lasso_model = Lasso(lambdas[i], fit_intercept=scaling, max_iter=2000)
    OLS_model = LinearRegression(fit_intercept=scaling)
    print("OLS")
    for n in range(N):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

        scores_ols = cross_validate(
            estimator=OLS_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_OLS[n] = np.mean(-scores_ols["test_score"])

        scores_Ridge = cross_validate(
            estimator=Ridge_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_Ridge[n] = np.mean(-scores_Ridge["test_score"])

        scores_Lasso = cross_validate(
            estimator=Lasso_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_Lasso[n] = np.mean(-scores_Lasso["test_score"])

    plt.plot(errors_OLS, label="OLS")
    plt.plot(errors_Ridge, label="Ridge")
    plt.plot(errors_Lasso, label="Lasso")
    plt.xlabel("Model Polynomial Degree")

    plt.ylim(0, 0.1)
    plt.tight_layout(h_pad=0.001)
    plt.legend()

    plt.title(f"lambda = {lambdas[i]:.5}")

plt.show()
