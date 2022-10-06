from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression
from imageio import imread

# Our own library of functions
from utils import *
# Load the terrain
z_terrain1 = imread("../data/tiny_SRTM_data_Norway_1.tif")
x_terrain1 = np.arange(z_terrain1.shape[0])
y_terrain1 = np.arange(z_terrain1.shape[1])
x1, y1 = np.meshgrid(x_terrain1, y_terrain1, indexing="ij")

z_terrain2 = imread("../data/SRTM_data_Norway_2.tif")
x_terrain2 = np.arange(z_terrain2.shape[0])
y_terrain2 = np.arange(z_terrain2.shape[1])
x2, y2 = np.meshgrid(x_terrain2, y_terrain2, indexing="ij")

# Show the terrain
plt.plot()
plt.subplot(121)
plt.title("Terrain over Norway 1")
plt.imshow(z_terrain1)
plt.xlabel("Y")
plt.ylabel("X")

plt.subplot(122)
plt.title("Terrain over Norway 2")
plt.imshow(z_terrain2)
plt.xlabel("Y")
plt.ylabel("X")
plt.show()

np.random.seed(42069)
N = 15
K = 10
bootstraps = 100
scaling = True

np.random.seed(42069)

def task_d(x, y, z, N, K, bootstraps, scaling):
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    X, X_train, X_test, z, z_train, z_test = minmax_dataset(
        X, X_train, X_test, z, z_train, z_test
    )

    z = z.ravel()

    OLS_model = LinearRegression(fit_intercept=False)
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
    plt.ylabel("MSE")
    plt.xlabel("Polynomial Degree")
    plt.title(
        f"MSE by Resampling Method, with scaling={scaling}, n={N}, k={K}, bootstraps={bootstraps}"
    )

    plt.legend()
    plt.show()

task_d(x1, y1, z_terrain1, N, K, bootstraps, scaling)