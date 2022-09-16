from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

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

def bootstrap(X, X_train, X_test, z_train, z_test, N, bootstraps *, scaling=False):
    z_preds = np.empty((z_test.shape[0], bootstraps))

    for i in range(N):
        for i in range(bootstraps):
            X_, z_ = resample(X_train, z_train)
            _, _, _, _, _, _, z_pred_ =  linear_regression(X, X_train, X_test, z_train, z_test, N, *, scaling=False)
            z_preds[:,i] = z_pred_

    return z_preds

def bias_variance(z_preds, z_test):


def preprocess(x, y, z, N, test_size):
    X = create_X(x, y, N)
    zflat = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, zflat, test_size=test_size)

    return X, X_train, X_test, z_train, z_test

def linear_regression(X, X_train, X_test, z_train, z_test, *, scaling=False):
    L = X_train.shape[1]

    if scaling:
        X_train = X_train[:,1:L]
        X_test = X_test[:,1:L]
        z_train_mean = np.mean(z_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        beta = np.linalg.pinv(X_train - X_train_mean.T @ X_train - X_train_mean) @ X_train - X_train_mean.T @ (z_train - z_train_mean)
        intercept = np.mean(z_train_mean - X_train_mean @ beta)
        z_pred_train = X_train @ beta + intercept
        z_pred_test = X_test @ beta + intercept
    else:
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
        z_pred_train = X_train @ beta
        z_pred_test = X_test @ beta

    return beta, z_pred_train, z_pred_test

def linreg_to_N(X, X_train, X_test, z_train, z_test, N, *, scaling=False):
    L = X_train.shape[1]

    betas = np.zeros((L,N+1))
    z_preds_train = np.empty((z_train.shape[0],L))
    z_preds_test = np.empty((z_test.shape[0],L))

    # if we scale we do not include the intercept coloumn
    if scaling:
        X_train = X_train[:,1:L]
        X_test = X_test[:,1:L]
        betas = np.zeros((L-1,N+1))
        z_train_mean = np.mean(z_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)

    for n in range(N+1):
        l = int((n+1)*(n+2)/2) # Number of elements in beta
        if scaling:
            l -= 1

        beta, z_pred_train, z_pred_test = linear_regression(X[:,0:l], X_train[:,0:l], X_test[:,0:l], z_train, z_test, scaling=False)

        betas[0:len(beta),n] = beta
        z_preds_test[:,n] = z_pred_test
        z_preds_train[:,n] = z_pred_train

    return betas, z_preds_test, z_preds_train

def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[:,n] = MSE(z_preds[:,n], z)
        R2s[:,n] = R2(z_preds[:,n], z)

    return MSEs, R2s

