"""
task g: compares resampling methods (bootstrap, cross validation and scikit cross validation) for ridge model
"""
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate, KFold

# Our own library of functions
from utils import *

np.random.seed(42069)

# Parameters
K = 20
bootstraps = 100
kfolds = KFold(n_splits=K, shuffle=True)

# if true, plot only lambda that gives lowest MSE
# if false, plot range of different lambdas
plot_only_best_lambda = True
lambdas = np.logspace(-12, -4, 6)

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

# plot only the gridsearched lambda
if plot_only_best_lambda:
    ridge = Ridge(fit_intercept=False)
    lam, _, _ = find_best_lambda(X, z, ridge, lambdas, N, K)
    lambdas = [lam]

# run through different values of lambda
for i in range(len(lambdas)):
    if not plot_only_best_lambda:
        plt.subplot(321 + i)
        plt.suptitle(f"MSE by polynomial degree for resampling methods Ridge")

    # arrays for MSE scores
    errors_cv = np.zeros(N+1)
    errors_cv_scikit = np.zeros(N+1)
    errors_boot = np.zeros(N+1)

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
    for n in range(N+1):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # number of elements in beta
        z_preds_test, _ = bootstrap(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            z_test,
            bootstraps,
            centering=centering,
            model=model,
            lam=lambdas[i],
        )

        error, _ = scores(z_test, z_preds_test)
        errors_boot[n] = np.mean(error)

    # plot subplots
    plt.plot(errors_boot, label="bootstrap")
    plt.plot(errors_cv, label="cross validation implementation")
    plt.plot(errors_cv_scikit, label="cross validation scikit learn")
    plt.ylabel("MSE score")
    plt.xlabel("Polynomial degree (N)")
    if plot_only_best_lambda:
        plt.title(
            f"MSE by polynomial degree for resampling methods \n for optimal lambda = {lambdas[i]} Ridge"
        )
    else:
        plt.title(f"lambda = {lambdas[i]:.5}")
    plt.legend()

plt.show()
