"""
task g: calculate and plot ridge beta value over model complexity
"""
from imageio import imread

# Our own library of functions
from utils import *

np.random.seed(42069)

# read in data
z_terrain1 = imread("../data/tiny_SRTM_data_Norway_1.tif")
x_terrain1 = np.arange(z_terrain1.shape[0])
y_terrain1 = np.arange(z_terrain1.shape[1])
x1, y1 = np.meshgrid(x_terrain1, y_terrain1, indexing="ij")

z_terrain2 = imread("../data/SRTM_data_Norway_2.tif")
x_terrain2 = np.arange(z_terrain2.shape[0])
y_terrain2 = np.arange(z_terrain2.shape[1])
x2, y2 = np.meshgrid(x_terrain2, y_terrain2, indexing="ij")

# Highest order polynomial we fit with
N = 30
lambdas = np.logspace(-15, -9, 6)
lambdas[-1] = 10000000000
betas_to_plot = 5

def ridge_betas(x, y, z, N, lambdas, betas_to_plot):
    # split data into train and test
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # normalize data
    X, X_train, X_test, z, z_train, z_test = normalize_task_g(
        X, X_train, X_test, z, z_train, z_test
    )

    # calculate beta values
    for i in range(len(lambdas)):
        plt.subplot(321 + i)
        betas, z_preds_train, z_preds_test, _ = linreg_to_N(
            X,
            X_train,
            X_test,
            z_train,
            z_test,
            N,
            model=ridge,
            lam=lambdas[i],
        )

    # plot beta values
        if betas_to_plot <= betas.shape[0]:
            for col in range(betas_to_plot):
                plt.plot(betas[col, :], label=f"beta{col}")

        plt.title(f"Beta progression for lambda = {lambdas[i]:.5}")
        plt.legend()

    plt.show()

ridge_betas(x1, y1, z_terrain1, N, lambdas, betas_to_plot)