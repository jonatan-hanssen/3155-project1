from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import *

# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

fig = plt.figure(figsize=plt.figaspect(0.5))


# print(f"{x=}{y=}")
N = 20
z = z + 0.1*np.random.standard_normal(z.shape)
betas, MSE_train, MSE_test, R2_train, R2_test = linear_regression(x,y,z, N)

X = create_X(x,y,N)

# SHOW PRED -----------------------
z_pred = X @ betas[:,N-1]

ax = fig.add_subplot(1,2,1,projection='3d')
# print(f"{z=} {z=}")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Franke Function')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


# Plot the surface.
ax = fig.add_subplot(1,2,2,projection='3d')
# print(f"{z=} {z=}")
# Plot the surface.
surf = ax.plot_wireframe(x, y, np.reshape(z_pred, z.shape), cstride=1, rstride=1)
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Polynomial fit of Franke Function')
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# np.set_printoptions(suppress=True)
# roundbetas = np.round(betas,4)
# print(f"{roundbetas=}")


plt.subplot(221)
plt.plot(betas[0,:],label="beta0")
plt.plot(betas[1,:],label="beta1")
plt.plot(betas[2,:],label="beta2")
plt.plot(betas[3,:],label="beta3")
plt.plot(betas[4,:],label="beta4")
plt.plot(betas[5,:],label="beta5")
plt.legend()

plt.subplot(222)

plt.plot(MSE_train, label="train")
plt.plot(MSE_test, label="test")
plt.legend()
plt.title("MSE scores")

plt.subplot(223)
plt.plot(R2_train, label="train")
plt.plot(R2_test, label="test")
plt.legend()
plt.title("R2 scores")


plt.show()

# linear_regression(x,y,z)


