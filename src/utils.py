from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from typing import Tuple, Callable


def FrankeFunction(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


# debug function
def SkrankeFunction(x, y):
    return x**2 + y**2


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)

    return X


def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n


def OLS(
    X_train: np.ndarray,
    z_train: np.ndarray,
):
    beta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ z_train

    return beta


def ridge(X_train, z_train, lam, *, scaling=False):
    L = X_train.shape[1]

    beta = np.linalg.pinv(X_train.T @ X_train + lam * np.eye(L)) @ X_train.T @ z_train

    return beta


def bootstrap(
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    bootstraps: int,
    *,
    scaling: bool = False,
    model: Callable = OLS,
    lam: float = 0,
):
    z_preds_train = np.empty((z_train.shape[0], bootstraps))
    z_preds_test = np.empty((z_test.shape[0], bootstraps))

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        _, z_pred_train, z_pred_test, _ = evaluate_model(
            X, X_, X_test, z_, model, lam=lam, scaling=scaling
        )
        z_preds_train[:, i] = z_pred_train
        z_preds_test[:, i] = z_pred_test

    return z_preds_test


def crossval(
    X: np.ndarray,
    z: np.ndarray,
    K: int,
    *,
    scaling: bool = False,
    model=OLS,
    lam: float = 0,
):
    chunksize = X.shape[0] // K

    errors = np.zeros(K)
    X, z = resample(X, z)

    for k in range(K):
        if k == K - 1:
            # if we are on the last, take all thats left
            X_test = X[k * chunksize :, :]
            z_test = z[k * chunksize :]
        else:
            X_test = X[k * chunksize : (k + 1) * chunksize, :]
            z_test = z[k * chunksize : (k + 1) * chunksize :]

        X_train = np.delete(
            X,
            [i for i in range(k * chunksize, k * chunksize + X_test.shape[0])],
            axis=0,
        )
        z_train = np.delete(
            z,
            [i for i in range(k * chunksize, k * chunksize + z_test.shape[0])],
            axis=0,
        )

        _, _, z_pred_test, _ = evaluate_model(
            X,
            X_train,
            X_test,
            z_train,
            model,
            lam=lam,
            scaling=scaling,
        )
        errors[k] = MSE(z_test, z_pred_test)

    return np.mean(errors)


def bias_variance(z_test: np.ndarray, z_preds_test: np.ndarray):
    MSEs, _ = scores(z_test, z_preds_test)
    error = np.mean(MSEs)
    bias = np.mean((z_test - np.mean(z_preds_test, axis=1, keepdims=True).flatten()) ** 2)
    variance = np.mean(np.var(z_preds_test, axis=1, keepdims=True))

    return error, bias, variance


def preprocess(x: np.ndarray, y: np.ndarray, z: np.ndarray, N, test_size):
    X = create_X(x, y, N)

    zflat = np.ravel(z)
    X_train, X_test, z_train, z_test = train_test_split(X, zflat, test_size=test_size)

    return X, X_train, X_test, z_train, z_test


def evaluate_model(
    X,
    X_train,
    X_test,
    z_train,
    model,
    *,
    lam: float = 0,
    scaling: bool = False,
):
    if isinstance(model, Callable):
        intercept = 0
        if scaling:
            X_train = X_train[:, 1:]
            X_test = X_test[:, 1:]
            X = X[:, 1:]
            z_train_mean = np.mean(z_train, axis=0)
            X_train_mean = np.mean(X_train, axis=0)

            if model.__name__ == "OLS":
                beta = model((X_train - X_train_mean), (z_train - z_train_mean))

            elif model.__name__ == "ridge":
                beta = model((X_train - X_train_mean), (z_train - z_train_mean), lam)

            intercept = np.mean(z_train_mean - X_train_mean @ beta)

        else:
            if model.__name__ == "OLS":
                beta = model(X_train, z_train)

            elif model.__name__ == "ridge":
                beta = model(
                    X_train,
                    z_train,
                    lam,
                )
        # intercept is zero if no scaling
        z_pred_train = X_train @ beta + intercept
        z_pred_test = X_test @ beta + intercept
        z_pred = X @ beta + intercept

    # presumed scikit model
    else:
        intercept = 0
        if scaling:
            # if width is 1, simply return the intercept
            if X_train.shape[1] == 1:
                beta = np.zeros(1)
                intercept = np.mean(z_train, axis=0)
                z_pred_train = np.ones(X_train.shape[0]) * intercept
                z_pred_test = np.ones(X_test.shape[0]) * intercept
                z_pred = np.ones(X.shape[0]) * intercept

                return beta, z_pred_train, z_pred_test, z_pred

            X_train = X_train[:, 1:]
            X_test = X_test[:, 1:]
            X = X[:, 1:]
            z_train_mean = np.mean(z_train, axis=0)
            X_train_mean = np.mean(X_train, axis=0)

            model.fit((X_train - X_train_mean), (z_train - z_train_mean))
            beta = model.coef_
            intercept = np.mean(z_train_mean - X_train_mean @ beta)
        else:
            model.fit(X_train, z_train)

        beta = model.coef_
        z_pred = model.predict(X) + intercept
        z_pred_train = model.predict(X_train) + intercept
        z_pred_test = model.predict(X_test) + intercept

    return beta, z_pred_train, z_pred_test, z_pred


def normalize_task_g(X, X_train, X_test, z, z_train, z_test):
    x_scaler = StandardScaler()
    z_scaler = StandardScaler()

    x_scaler.fit(X_train)
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    X = x_scaler.transform(X)

    z_shape = z.shape

    # make all zeds into 1 dimensional arrays for standardscaler
    z_train = z_train.reshape((z_train.shape[0], 1))
    z_test = z_test.reshape((z_test.shape[0], 1))
    z = z.ravel().reshape((z.ravel().shape[0], 1))

    z_scaler.fit(z_train)
    z_train = np.ravel(z_scaler.transform(z_train))
    z_test = np.ravel(z_scaler.transform(z_test))
    z = np.ravel(z_scaler.transform(z))
    z = z.reshape(z_shape)

    return X, X_train, X_test, z, z_train, z_test


def linreg_to_N(
    X: np.ndarray,
    X_train: np.ndarray,
    X_test: np.ndarray,
    z_train: np.ndarray,
    z_test: np.ndarray,
    N: int,
    *,
    scaling: bool = False,
    model: Callable = OLS,
    lam: float = 0,
    bootstrap: bool = False,
):
    L = X_train.shape[1]

    betas = np.zeros((L, N + 1))
    z_preds_train = np.empty((z_train.shape[0], N + 1))
    z_preds_test = np.empty((z_test.shape[0], N + 1))
    z_preds = np.empty((X.shape[0], N + 1))

    for n in range(N + 1):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        beta, z_pred_train, z_pred_test, z_pred = evaluate_model(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            model,
            lam=lam,
            scaling=scaling,
        )

        betas[0 : len(beta), n] = beta
        z_preds_test[:, n] = z_pred_test
        z_preds_train[:, n] = z_pred_train
        z_preds[:, n] = z_pred

    return betas, z_preds_train, z_preds_test, z_preds


def scores(z, z_preds):
    N = z_preds.shape[1]
    MSEs = np.zeros((N))
    R2s = np.zeros((N))

    for n in range(N):
        MSEs[n] = MSE(z, z_preds[:, n])
        R2s[n] = R2(z, z_preds[:, n])

    return MSEs, R2s
