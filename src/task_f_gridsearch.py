from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline


# Our own library of functions
from utils import *

np.random.seed(42069)
# Make data.
x = np.arange(0, 1, 0.15)
y = np.arange(0, 1, 0.15)
# x = np.arange(0, 1, 0.05)
# y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)

# Highest order polynomial we fit with
N = 15
K = 10

# Do the linear_regression
z += 0.15 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)
z = z.ravel()

lambdas = np.logspace(-12, 0, 25)
ridge = Ridge(fit_intercept=True)
# ridge = LinearRegression()
# ridge = Lasso(fit_intercept=False)

ridge_reg = GridSearchCV(
    ridge,
    param_grid={"alpha": list(lambdas)},
    # param_grid={"fit_intercept": [False, True]},
    scoring="neg_mean_squared_error",
    cv=K,
)
degrees = np.arange(N)
best_degree = 0
best_lambda = 0
best_score = 10**10

best_lambdas = np.zeros(N)
best_scores = np.zeros(N)

for n in range(N):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    ridge_reg.fit(X[:, :l], z)
    best_scores[n] = -ridge_reg.best_score_
    best_lambdas[n] = ridge_reg.best_params_["alpha"]

    if -ridge_reg.best_score_ < best_score:
        best_score = -ridge_reg.best_score_
        best_lambda = ridge_reg.best_params_["alpha"]
        best_degree = n

plt.plot(best_scores, label="best score")
plt.plot(best_lambdas, "o", label="best lambda")
# plt.ylim(0, 0.2)
plt.legend()
plt.show()


print(f"{best_degree=}")
print(f"{best_score=}")
print(f"{best_lambda=}")
print(f"{best_lambdas=}")
print(f"{best_scores=}")


# linear_regression(x,y,z)
