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

def SkrankeFunction(x,y):
    return x**2 + y**2

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

def bootstrap(X, X_train, X_test, z_train, z_test, bootstraps, *, scaling=False):
    z_preds = np.zeros((z_test.shape[0], bootstraps))

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        _, _, z_pred_test, _ =  linear_regression(X, X_, X_test, z_, scaling=scaling)
        z_preds[:,i] = z_pred_test
    return z_preds

def crossval(X, z, K, *, scaling=False):
    chunksize = X.shape[0]//K

    errors = np.zeros(K)
    X, z = resample(X,z)

    for k in range(K):
        if k == K-1:
            # if we are on the last, take all thats left
            X_test = np.zeros((X.shape[0] - K*chunksize,X.shape[1]))
            X_test = X[k*chunksize:,:]
            z_test = z[k*chunksize:]
        else:
            X_test = X[k*chunksize:(k+1)*chunksize,:]
            z_test = z[k*chunksize:(k+1)*chunksize:]

        X_train = np.delete(X, [i for i in range(k*chunksize,k*chunksize+X_test.shape[0])], axis=0)
        z_train = np.delete(z, [i for i in range(k*chunksize,k*chunksize+z_test.shape[0])], axis=0)

        _, _, z_pred_test, _ = linear_regression(X, X_train, X_test, z_train, scaling=scaling)
        errors[k] = MSE(z_test, z_pred_test)

    return np.mean(errors)

def bias_variance(z_test, z_preds):
    MSEs, _ = scores(z_test, z_preds)
    error = np.mean(MSEs)
    bias = np.mean((z_test - np.mean(z_preds, axis=1, keepdims=True).flatten())**2)
    variance = np.mean(np.var(z_preds, axis=1, keepdims=True))

    return error, bias, variance

def preprocess(x, y, z, N, test_size):
    X = create_X(x, y, N)
    zflat = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, zflat, test_size=test_size)

    return X, X_train, X_test, z_train, z_test

def linear_regression(X, X_train, X_test, z_train, *, scaling=False):
    L = X_train.shape[1]

    if scaling:
        X_train = X_train[:,1:L]
        X_test = X_test[:,1:L]
        X = X[:,1:L]
        z_train_mean = np.mean(z_train, axis=0)
        X_train_mean = np.mean(X_train, axis=0)
        beta = np.linalg.pinv((X_train - X_train_mean).T @ (X_train - X_train_mean)) @ (X_train - X_train_mean).T @ (z_train - z_train_mean)
        intercept = np.mean(z_train_mean - X_train_mean @ beta)
        z_pred_train = X_train @ beta + intercept
        z_pred_test = X_test @ beta + intercept
        z_pred = X @ beta + intercept
    else:
        beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train
        z_pred_train = X_train @ beta
        z_pred_test = X_test @ beta
        z_pred = X @ beta

    return beta, z_pred_train, z_pred_test, z_pred

def linreg_to_N(X, X_train, X_test, z_train, z_test, N, *, scaling=False):
    L = X_train.shape[1]

    betas = np.zeros((L,N+1))
    z_preds_train = np.empty((z_train.shape[0],N+1))
    z_preds_test = np.empty((z_test.shape[0],N+1))
    z_preds = np.empty((X.shape[0],N+1))

    for n in range(N+1):
        l = int((n+1)*(n+2)/2) # Number of elements in beta

        beta, z_pred_train, z_pred_test, z_pred = linear_regression(X[:,0:l], X_train[:,0:l], X_test[:,0:l], z_train, scaling=scaling)

        betas[0:len(beta),n] = beta
        z_preds_test[:,n] = z_pred_test
        z_preds_train[:,n] = z_pred_train
        z_preds[:,n] = z_pred

    return betas, z_preds_train, z_preds_test, z_preds

def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[n] = MSE(z, z_preds[:,n])
        R2s[n] = R2(z, z_preds[:,n])

    return MSEs, R2s

