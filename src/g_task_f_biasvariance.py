"""
task g: performs bias-variance tradeoff for lasso regression using the scikit learn
        lasso model
"""
from sklearn.linear_model import Lasso
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
N = 8
bootstraps = 10
lambdas = np.logspace(-6, -1, 6)

# todo use only gridsearched lambda
def lasso_bias_variance(x, y, z, N, bootstraps, lambdas):
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    X, X_train, X_test, z, z_train, z_test = normalize_task_g(
         X, X_train, X_test, z, z_train, z_test
     )

    # loop through different lambda values
    for i in range(len(lambdas)):
        plt.subplot(321 + i)
        plt.suptitle(f"Bias variance tradeoff for lasso regression")

        # model under testing
        model_Lasso = Lasso(lambdas[i], max_iter=100, fit_intercept=False)

        # arrays for bias-variance
        errors = np.zeros(N)
        biases = np.zeros(N)
        variances = np.zeros(N)

        # for polynomial degree
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
                model=model_Lasso,
                lam=lambdas[i],
            )

            # calculate bias-variance
            error, bias, variance = bias_variance(z_test, z_preds)
            errors[n] = error
            biases[n] = bias
            variances[n] = variance

        # plot subplots
        plt.plot(errors, "g--", label="MSE test")
        plt.plot(biases, label="bias")
        plt.plot(variances, label="variance")
        # plt.ylim(0, 0.25)
        plt.xlabel("Polynomial Degree")
        plt.tight_layout(h_pad=0.001)

        plt.title(f"lambda = {lambdas[i]:.5}")

        plt.legend()

    plt.show()

lasso_bias_variance(x1, y1, z_terrain1, N, bootstraps, lambdas)