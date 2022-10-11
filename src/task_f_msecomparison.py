"""
task f: Plots MSE comparison between OLS, ridge and lasso with cross validation for real and synthetic data.
"""
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_validate, KFold

# Our own library of functions
from utils import *

# parameters
K = 10
lambdas = np.logspace(-12, -4, 6)

kfolds = KFold(n_splits=K)

(
    betas_to_plot,
    N,
    X,
    X_train,
    X_test,
    z,
    z_train,
    z_test,
    centering,
    x,
    y,
    z,
) = read_from_cmdline()

errors_OLS = np.zeros(N)
errors_Ridge = np.zeros(N)
errors_Lasso = np.zeros(N)

min_error_OLS = np.inf
min_error_Ridge = np.inf
min_error_Lasso = np.inf

best_poly_OLS = 0
best_poly_Ridge = 0
best_poly_Lasso = 0

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

    if min(min_error_OLS, np.min(scores_ols)) == np.min(scores_ols):
        min_error_OLS = np.min(scores_ols)
        best_poly_OLS = np.argmin(scores_ols)
    if min(min_error_Ridge, np.min(scores_Ridge)) == np.min(scores_Ridge):
        min_error_Ridge = np.min(scores_Ridge)
        best_poly_Ridge = np.argmin(scores_Ridge)
    if min(min_error_Lasso, np.min(scores_Lasso)) == np.min(scores_Lasso):
        min_error_Lasso = np.min(scores_Lasso)
        best_poly_Lasso = np.argmin(scores_Lasso)
    # plot
    plt.plot(errors_OLS, "r--", label="OLS")
    plt.plot(errors_Ridge, "b--", label="Ridge")
    plt.plot(errors_Lasso, "g--", label="Lasso")
    plt.ylabel("MSE score")
    plt.xlabel("Polynomial degree (N)")
    plt.title(f"lambda = {lambdas[i]:.5}")
    plt.tight_layout(h_pad=0.001)
    plt.legend()

print(f"Minimal MSE_test value for OLS = {min_error_OLS} for N = {best_poly_OLS}")
print(f"Minimal MSE_test value for Ridge = {min_error_Ridge} for N = {best_poly_Ridge}")
print(f"Minimal MSE_test value for Lasso = {min_error_Lasso} for N = {best_poly_Lasso}")
plt.show()
