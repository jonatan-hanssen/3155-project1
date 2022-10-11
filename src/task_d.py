"""
task d (and task g): compare resampling method MSE for OLS over N polynomial degrees
"""
from sklearn.model_selection import cross_validate, KFold

# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
K = 10
bootstraps = 100

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

z = z.ravel()

# model under testing
OLS_model = OLS

# scikit comparison model
OLS_scikit = LinearRegression(fit_intercept=False)
kfolds = KFold(n_splits=K, shuffle=True)

# cross val results
errors_cv = np.zeros(N + 1)
errors_cv_scikit = np.zeros(N + 1)

# bootstrap results
errors_boot = np.zeros(N + 1)
biases_boot = np.zeros(N + 1)
variances_boot = np.zeros(N + 1)

# run cross val
print("Cross")
for n in range(N):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

    # own implementation
    errors_cv[n] = crossval(X[:, :l], z, K, centering=centering, model=OLS_model)

    # scikit
    error_scikit = cross_validate(
        estimator=OLS_scikit,
        X=X[:, :l],
        y=z,
        scoring="neg_mean_squared_error",
        cv=kfolds,
    )
    errors_cv_scikit[n] = np.mean(-error_scikit["test_score"])

# run bootstrap
print("Boot")
for n in range(N + 1):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

    z_preds, _ = bootstrap(
        X[:, :l],
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        centering=centering,
        model=OLS_model,
    )

    # bias variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds)
    errors_boot[n] = error
    biases_boot[n] = bias
    variances_boot[n] = variance

# plot
plt.plot(errors_boot, label="bootstrap")
plt.plot(errors_cv, label="cross validation implementation")
plt.plot(errors_cv_scikit, label="cross validation scikit learn")
plt.ylabel("MSE score", size=15)
plt.xlabel("Polynomial degree (N)", size=15)
plt.title(f"MSE by Resampling Method", size=18)
plt.legend(prop={"size": 10})

plt.show()
