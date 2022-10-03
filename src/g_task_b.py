from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imageio import imread
from sklearn.preprocessing import StandardScaler

# Our own library of functions
from utils import *

np.random.seed(42069)

# Load the terrain
z_terrain1 = np.asarray(imread("../data/tiny_SRTM_data_Norway_1.tif"), dtype=float)
x_terrain1 = np.arange(z_terrain1.shape[0])
y_terrain1 = np.arange(z_terrain1.shape[1])
x1, y1 = np.meshgrid(x_terrain1, y_terrain1, indexing="ij")

z_terrain2 = imread("../data/SRTM_data_Norway_2.tif")
# x_terrain2 = np.arange(z_terrain2.shape[0])
# y_terrain2 = np.arange(z_terrain2.shape[1])
# x2, y2 = np.meshgrid(x_terrain2, y_terrain2)

# Show the terrain
plt.plot()
plt.subplot(121)
plt.title("Terrain over Norway 1")
plt.imshow(z_terrain1, cmap="gray")
plt.xlabel("Y")
plt.ylabel("X")

plt.subplot(122)
plt.title("Terrain over Norway 2")
plt.imshow(z_terrain2, cmap="gray")
plt.xlabel("Y")
plt.ylabel("X")
plt.show()

# Highest order polynomial we fit with
N = 10
scaling = False


def task_b(x, y, z, N, scaling):
    # Do the linear_regression
    print(z)
    z += 0.05 * np.random.standard_normal(z.shape)
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.001)

    the_forbidden_scaler = StandardScaler()
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X = scaler.transform(X)

    the_forbidden_scaler.fit(z_train.reshape((z_train.shape[0], 1)))
    z_train = np.ravel(
        the_forbidden_scaler.transform(z_train.reshape((z_train.shape[0], 1)))
    )
    z_test = np.ravel(
        the_forbidden_scaler.transform(z_test.reshape((z_test.shape[0], 1)))
    )
    shape = z.shape
    z = np.ravel(
        the_forbidden_scaler.transform(z.ravel().reshape((z.ravel().shape[0], 1)))
    )
    z = z.reshape(shape)

    betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS
    )

    pred_map = z_preds[:, -1].reshape(z_terrain1.shape)

    print(f"{z=}")
    print(f"{z_preds[:, -1]=}")

    # Calculate scores OLS without resampling
    MSE_train, R2_train = scores(z_train, z_preds_train)
    MSE_test, R2_test = scores(z_test, z_preds_test)

    # ScikitLearn OLS for comparison
    OLS_scikit = LinearRegression(fit_intercept=scaling)
    # OLS_scikit.fit(X_train, z_train)

    # _, z_preds_train_sk, z_preds_test_sk, z_preds = linreg_to_N(
    #     X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_scikit
    # )

    # MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
    # MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

    # ------------ PLOTTING 3D -----------------------
    fig = plt.figure(figsize=plt.figaspect(0.3))

    # Subplot for Franke Function
    ax = fig.add_subplot(121, projection="3d")
    # Plot the surface.
    print(f"{x.shape=}")
    print(f"{y.shape=}")
    print(f"{z.shape=}")
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_title("Scaled terrain")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # Add a color bar which maps values to colors.

    # Subplot for the prediction
    # Plot the surface.
    ax = fig.add_subplot(122, projection="3d")
    # print(f"{z=} {z=}")
    # Plot the surface.
    surf = ax.plot_surface(
        x,
        y,
        np.reshape(z_preds[:, N], z.shape),
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    # Customize the z axis.
    # ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_title("Polynomial fit of scaled terrain")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    # np.set_printoptions(suppress=True)
    # roundbetas = np.round(betas,4)
    # print(f"{roundbetas=}")

    # ---------------- PLOTTING GRAPHS --------------
    plt.subplot(121)
    plt.title("Terrain 1")
    plt.imshow(z)
    plt.colorbar()

    plt.subplot(122)
    plt.title("Predicted terrain 1")
    plt.imshow(pred_map)
    plt.colorbar()
    plt.show()

    plt.subplot(221)
    plt.plot(betas[0, :], label="beta0")
    plt.plot(betas[1, :], label="beta1")
    plt.plot(betas[2, :], label="beta2")
    plt.plot(betas[3, :], label="beta3")
    plt.plot(betas[4, :], label="beta4")
    plt.plot(betas[5, :], label="beta5")
    plt.xlabel("Polynomial degree")
    plt.legend()
    plt.title("Beta progression")

    plt.subplot(222)

    plt.plot(MSE_train, label="train implementation")
    plt.plot(MSE_test, label="test implementation")
    # plt.plot(MSE_train_sk, "r--", label="train ScikitLearn")
    # plt.plot(MSE_test_sk, "g--", label="test ScikitLearn")
    plt.ylabel("MSE score")
    plt.xlabel("Polynomial degree")
    # plt.ylim(0, 0.1)
    plt.legend()
    plt.title("MSE scores over model complexity")

    plt.subplot(223)
    plt.plot(R2_train, label="train implementation")
    plt.plot(R2_test, label="test implementation")
    # plt.plot(R2_train_sk, "r--", label="train ScikitLearn")
    # plt.plot(R2_test_sk, "g--", label="test ScikitLearn")
    plt.ylabel("R2 score")
    plt.xlabel("Polynomial degree")
    # plt.ylim(-2, 1)
    plt.legend()
    plt.title("R2 scores over model complexity")

    plt.show()


# linear_regression(x,y,z)
task_b(x1, y1, z_terrain1, N, scaling)
# task_b(x2, y2, z_terrain2, N, scaling)
