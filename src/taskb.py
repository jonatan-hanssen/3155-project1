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
xv, yv = np.meshgrid(x,y)

fig = plt.figure()
ax = fig.gca(projection='3d')
zv = FrankeFunction(xv, yv)
z = FrankeFunction(x, y)
# Plot the surface.
surf = ax.plot_surface(xv, yv, zv, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


betas, MSE_train, MSE_test, R2_train, R2_test = linear_regression(x,y,z)

z += 0.1*np.random.standard_normal(len(z))
linear_regression(x,y,z)

