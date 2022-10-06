"""
task g: compares resampling methods (bootstrap, cross validation and scikit cross validation for ridge model
"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_validate, KFold
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

# highest order polynomial we fit with
N = 10
# amount of folds
K = 10
bootstraps = 100

# todo change plot to only gridsearched lambda
def ridge_mse_compare_resampling(x, y, z, N, K, bootstraps):
    kfolds = KFold(n_splits=K)

    # split into training and test
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # normalize data based on train
    X, X_train, X_test, z, z_train, z_test = minmax_dataset(
        X, X_train, X_test, z, z_train, z_test
    )

    z = z.ravel()

    # run through different values of lambda
    lambdas = np.logspace(-12, -4, 6)
    for i in range(len(lambdas)):
        plt.subplot(321 + i)
        plt.suptitle(f"MSE by polynomial degree for different resampling methods")

        # arrays for MSE scores
        errors_cv = np.zeros(N)
        errors_cv_scikit = np.zeros(N)
        errors_boot = np.zeros(N)

        # scikit model under testing
        Ridge_model = Ridge(lambdas[i], fit_intercept=False)

        # implemented model under testing
        model = ridge

        # calculate cross validation for own implementation and scikit error
        print("Running Cross Validation")
        for n in range(N):
            print(n)
            l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
            errors_cv[n] = crossval(X[:, :l], z, K)

            error_scikit = cross_validate(
                estimator=Ridge_model,
                X=X[:, :l],
                y=z,
                scoring="neg_mean_squared_error",
                cv=kfolds,
            )
            errors_cv_scikit[n] = np.mean(-error_scikit["test_score"])

        # calculate bootstrap error
        print("Running Bootstrap")
        for n in range(N):
            print(n)
            l = int((n + 1) * (n + 2) / 2)  # number of elements in beta
            z_preds = bootstrap(
                X[:, :l],
                X_train[:, :l],
                X_test[:, :l],
                z_train,
                z_test,
                bootstraps,
                model=model,
                lam=lambdas[i],
            )

            error, _, _ = bias_variance(z_test, z_preds)
            errors_boot[n] = error

        # plot subplots
        plt.plot(errors_boot, label="bootstrap")
        plt.plot(errors_cv, label="cross validation implementation")
        plt.plot(errors_cv_scikit, label="cross validation skicit learn")
        plt.xlabel("Model Polynomial Degree")
        plt.title(f"lambda = {lambdas[i]:.5}")
        plt.legend()

    plt.show()


ridge_mse_compare_resampling(x1, y1, z_terrain1, N, K, bootstraps)
