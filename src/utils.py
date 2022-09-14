from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

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

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def linear_regression(x, y, z):
    print(x.shape, y.shape, z.shape)
    MSE_train = np.zeros((5))
    MSE_test = np.zeros((5))
    R2_train = np.zeros((5))
    R2_test = np.zeros((5))
    betas = np.zeros((5))

    for n in range(5):
        X = create_X(x, y, n)

        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

        z_model = X_train @ beta
        z_model_test = X_test @ beta

        MSE_train[n] = MSE(z_train, z_model)
        MSE_test[n] = MSE(z_test, z_model_test)
        R2_train[n] = R2(z_train, z_model)
        R2_test[n] = R2(z_test, z_model_test)
        betas[n] = beta

    return betas, MSE_train, MSE_test, R2_train, R2_test


