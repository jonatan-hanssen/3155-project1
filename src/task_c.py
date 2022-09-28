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
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)
N = 15
bootstraps = 100

np.random.seed(42069)

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
        scaling=True,
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
