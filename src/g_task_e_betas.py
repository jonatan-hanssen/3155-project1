from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from imageio import imread

# Our own library of functions
from utils import *

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
# Make data.

# Highest order polynomial we fit with
N = 30

def task_e_betas(x, y, z, N):
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    X, X_train, X_test, z, z_train, z_test = normalize_task_g(
        X, X_train, X_test, z, z_train, z_test
    )

    lambdas = np.logspace(-15, -9, 6)
    lambdas[-1] = 10000000000
    for i in range(len(lambdas)):
        plt.subplot(321 + i)
        betas, z_preds_train, z_preds_test, _ = linreg_to_N(
            X,
            X_train,
            X_test,
            z_train,
            z_test,
            N,
            model=ridge,
            scaling=False,
            lam=lambdas[i],
        )
        for col in range(9):
            plt.plot(betas[col, :], label=f"beta{col}")

        plt.title(f"Beta progression for lambda = {lambdas[i]:.5}")
        plt.legend()

    plt.show()

task_e_betas(x1, y1, z_terrain1, N)