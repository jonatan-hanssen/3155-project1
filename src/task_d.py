from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression

# Our own library of functions
from utils import *

np.random.seed(42070)
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
z += 0.15 * np.random.standard_normal(z.shape)
# z = SkrankeFunction(x, y)
N = 15
K = 10
bootstraps = 100
scaling = False

np.random.seed(42069)

X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

z = z.ravel()


OLS_model = LinearRegression(fit_intercept=scaling)
kfolds = KFold(n_splits=K)

errors_cv = np.zeros(N)
errors_cv_scikit = np.zeros(N)

errors_boot = np.zeros(N)
biases_boot = np.zeros(N)
variances_boot = np.zeros(N)

print("Cross")
for n in range(N):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    errors_cv[n] = crossval(X[:, :l], z, K, scaling=scaling)

    error_scikit = cross_validate(
        estimator=OLS_model,
        X=X[:, :l],
        y=z,
        scoring="neg_mean_squared_error",
        cv=kfolds,
    )
    errors_cv_scikit[n] = np.mean(-error_scikit["test_score"])


print("Boot")
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
        scaling=scaling,
    )

    error, bias, variance = bias_variance(z_test, z_preds)
    errors_boot[n] = error
    biases_boot[n] = bias
    variances_boot[n] = variance

plt.plot(errors_boot, label="bootstrap")
plt.plot(errors_cv, label="cross validation implementation")
plt.plot(errors_cv_scikit, label="cross validation skicit learn")
plt.xlabel("Polynomial Degree")
plt.title(
    f"MSE by Resampling Method, with scaling={scaling}, n={N}, k={K}, bootstraps={bootstraps}"
)
plt.ylim(0, 0.1)

plt.legend()
plt.show()
