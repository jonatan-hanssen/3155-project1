"""
task f: Plots MSE comparison between OLS, ridge and lasso with cross validation for real and synthetic data.
"""
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_validate, KFold

# Our own library of functions
from utils import *

# parameters
N = 20
K = 10
noise = 0.05
kfolds = KFold(n_splits=K)
scaling = False
lambdas = np.logspace(-12, -4, 6)

errors_OLS = np.zeros(N)
errors_Ridge = np.zeros(N)
errors_Lasso = np.zeros(N)

# read in data
X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(N, scaling, noise)

# test different values of lambda
for i in range(len(lambdas)):
    z = z.ravel()
    plt.subplot(321 + i)
    plt.suptitle(f"MSE by polynomial degree for OLS, Ridge and Lasso regression")

    # models under testing
    Ridge_model = Ridge(lambdas[i], fit_intercept=False)
    Lasso_model = Lasso(lambdas[i], fit_intercept=False, max_iter=200)
    OLS_model = LinearRegression(fit_intercept=False)

    # cross validate, get error
    print("OLS")
    for n in range(N):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

        scores_ols = cross_validate(
            estimator=OLS_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_OLS[n] = np.mean(-scores_ols["test_score"])

        scores_Ridge = cross_validate(
            estimator=Ridge_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_Ridge[n] = np.mean(-scores_Ridge["test_score"])

        scores_Lasso = cross_validate(
            estimator=Lasso_model,
            X=X[:, :l],
            y=z,
            scoring="neg_mean_squared_error",
            cv=kfolds,
        )
        errors_Lasso[n] = np.mean(-scores_Lasso["test_score"])

    # plot
    plt.plot(errors_OLS, 'r--', label="OLS")
    plt.plot(errors_Ridge, 'b--', label="Ridge")
    plt.plot(errors_Lasso, 'g--', label="Lasso")
    plt.ylabel("MSE score")
    plt.xlabel("Polynomial degree (N)")
    plt.title(f"lambda = {lambdas[i]:.5}")
    plt.tight_layout(h_pad=0.001)
    plt.legend()

plt.show()