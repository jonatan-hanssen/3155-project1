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

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)
N = 15
bootstraps = 50

z += 0.15*np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

errors = np.zeros(N)
biases = np.zeros(N)
variances = np.zeros(N)

for n in range(N):
    print(n)
    z_preds = bootstrap(X, X_train, X_test, z_train, z_test, n, bootstraps)
    if n == 4:
        print(z_preds)

    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n] = error
    biases[n] = bias
    variances[n] = variance


bias = z_test - 0.43
bias = np.mean(bias**2)

print(z_test)

plt.plot(errors, label="error")
plt.plot(biases, label="biases")
plt.plot(variances, label="variances")
plt.legend()
plt.show()
