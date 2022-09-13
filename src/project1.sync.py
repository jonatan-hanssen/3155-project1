# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# Here we have the franke function

# %%

# imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# %%
fig = plt.figure()
ax = fig.gca(projection='3d')
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
xv, yv = np.meshgrid(x,y)

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y)
# Plot the surface.
surf = ax.plot_surface(xv, yv, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# %%
def create_X(x, y, n ):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)		# Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X
# %%
def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

# %%
MSE_arr_train = np.zeros((5))
MSE_arr_test = np.zeros((5))
R2_arr_train = np.zeros((5))
R2_arr_test = np.zeros((5))

for n in range(5):
    X = create_X(x, y, n)

    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
    z_model = X_train @ beta
    z_model_test = X_test @ beta
    MSE_arr_train[n] = MSE(z_train, z_model)
    MSE_arr_test[n] = MSE(z_test, z_model_test)
    R2_arr_train[n] = R2(z_train, z_model)
    R2_arr_test[n] = R2(z_test, z_model_test)
print(f"Training data; MSE: {MSE_arr_train}, R2: {R2_arr_train}")
print(f"Training data; MSE: {MSE_arr_test}, R2: {R2_arr_test}")
plt.plot(MSE_arr_train)
plt.plot(MSE_arr_test)
plt.title("MSE scores")
plt.show()
plt.plot(R2_arr_train)
plt.plot(R2_arr_test)
plt.title("R2 scores")
plt.show()
# %%


plt.plot(X_train @ beta)
plt.scatter(range(80), z_train)
plt.show()

y_model = X_test @ beta

MSE_test = MSE(z_test, y_model)
R2_test = R2(z_test, y_model)
print(f"Test data; MSE: {MSE_test}, R2: {R2_test}")
plt.plot(X_test @ beta)
plt.scatter(range(20), z_test)
plt.show()

# %%
scikit_linreg = LinearRegression(fit_intercept=False).fit(X_train, z_train)
scikit_model = scikit_linreg.predict(X_train) 
MSE_train = MSE(z_train, scikit_model)
R2_train = R2(z_train, scikit_model)
print(f"Training data; MSE: {MSE_train}, R2: {R2_train}")

plt.plot(scikit_model)
plt.scatter(range(80), z_train)
plt.show()

MSE_test = MSE(z_test, scikit_model)
R2_test = R2(z_test, scikit_model)
print(f"Test data; MSE: {MSE_test}, R2: {R2_test}")

plt.plot(scikit_model)
plt.scatter(range(20), z_test)
plt.show()


