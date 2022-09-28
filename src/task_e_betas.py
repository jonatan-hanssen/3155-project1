from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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
N = 30
lam = 0.001

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

lambdas = np.logspace(-4, 4, 6)
for i in range(len(lambdas)):
    plt.subplot(321 + i)
    betas, z_preds_train, z_preds_test, _ = linreg_to_N(
        X,
        X_train,
        X_test,
        z_train,
        z_test,
        N,
        scaling=True,
        model=ridge,
        lam=lambdas[i],
    )
    plt.plot(betas[0, :], label="beta0")
    plt.plot(betas[1, :], label="beta1")
    plt.plot(betas[2, :], label="beta2")
    plt.plot(betas[3, :], label="beta3")
    plt.plot(betas[4, :], label="beta4")
    plt.plot(betas[5, :], label="beta5")
    plt.title(f"Beta progression for lambda = {lambdas[i]:.5}")
    plt.legend()

# Calculate scores OLS without resampling
# MSE_train, R2_train = scores(z_train, z_preds_train)
# MSE_test, R2_test = scores(z_test, z_preds_test)

# ---------------- PLOTTING GRAPHS --------------

# plt.subplot(222)
#
# plt.plot(MSE_train, label="train")
# plt.plot(MSE_test, label="test")
# plt.xlabel("Polynomial degree")
# plt.legend()
# plt.title("MSE scores")
#
# plt.subplot(223)
# plt.plot(R2_train, label="train")
# plt.plot(R2_test, label="test")
# plt.xlabel("Polynomial degree")
# plt.legend()
# plt.title("R2 scores")


plt.show()

# linear_regression(x,y,z)
