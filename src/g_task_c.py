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

# Load the terrain
z_terrain1 = np.asarray(imread("../data/tiny_SRTM_data_Norway_1.tif"), dtype=float)
x_terrain1 = np.arange(z_terrain1.shape[0])
y_terrain1 = np.arange(z_terrain1.shape[1])
x1, y1 = np.meshgrid(x_terrain1, y_terrain1, indexing="ij")

z_terrain2 = imread("../data/SRTM_data_Norway_2.tif")
x_terrain2 = np.arange(z_terrain2.shape[0])
y_terrain2 = np.arange(z_terrain2.shape[1])
x2, y2 = np.meshgrid(x_terrain2, y_terrain2)

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

N = 15
bootstraps = 100
scaling = False

np.random.seed(42069)

def task_c(x, y, z, N, scaling, bootstraps):
    z += 0.15 * np.random.standard_normal(z.shape)
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2) 

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
            scaling=scaling,
        )

        error, bias, variance = bias_variance(z_test, z_preds)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    plt.plot(errors, label="error")
    plt.plot(biases, label="biases")
    plt.plot(variances, label="variances")
    plt.xlabel("Polynomial degree")
    plt.title("Bias-variance tradeoff over model complexity")
    plt.legend()
    plt.show()

task_c(x1, y1, z_terrain1, N, scaling ,bootstraps)
