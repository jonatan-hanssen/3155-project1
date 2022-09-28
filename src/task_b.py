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
N = 15
scaling = False

# Do the linear_regression
z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)
betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS
)
print(f"{betas=}")
# Calculate scores OLS without resampling
MSE_train, R2_train = scores(z_train, z_preds_train)
MSE_test, R2_test = scores(z_test, z_preds_test)

# ScikitLearn OLS for comparison
OLS_model = LinearRegression(fit_intercept=scaling)
OLS_model.fit(X_train, z_train)

_, z_preds_train_sk, z_preds_test_sk, _ = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_model
)
print(f"{z_preds_test_sk.shape=}")
print(f"{z_preds_train_sk.shape=}")

MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

# ------------ PLOTTING 3D -----------------------
fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for Franke Function
ax = fig.add_subplot(1, 2, 1, projection="3d")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Franke Function")
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


# Subplot for the prediction
# Plot the surface.
ax = fig.add_subplot(1, 2, 2, projection="3d")
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
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Polynomial fit of Franke Function")
fig.colorbar(surf, shrink=0.5, aspect=5)

# Subplot with overlayed prediction
# ax = fig.add_subplot(1,3,3,projection='3d')
# # print(f"{z=} {z=}")
# # Plot the surface.
# surf = ax.plot_wireframe(x, y, np.reshape(z_pred, z.shape), cstride=1, rstride=1)
# surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# # Customize the z axis.
# ax.set_zlim(-0.10, 1.40)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.set_title('Fit overlayed on Franke Function')

plt.show()

# np.set_printoptions(suppress=True)
# roundbetas = np.round(betas,4)
# print(f"{roundbetas=}")


# ---------------- PLOTTING GRAPHS --------------
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
plt.plot(MSE_train_sk, label="train ScikitLearn")
plt.plot(MSE_test_sk, label="test ScikitLearn")
plt.ylabel("MSE score")
plt.xlabel("Polynomial degree")
plt.legend()
plt.title("MSE scores over model complexity")

plt.subplot(223)
plt.plot(R2_train, label="train implementation")
plt.plot(R2_test, label="test implementation")
plt.plot(R2_train, label="train ScikitLearn")
plt.plot(R2_test, label="test ScikitLearn")
plt.ylabel("R2 score")
plt.xlabel("Polynomial degree")
plt.legend()
plt.title("R2 scores over model complexity")

plt.show()

# linear_regression(x,y,z)
