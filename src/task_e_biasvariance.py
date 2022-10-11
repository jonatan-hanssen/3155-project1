"""
task e (and task g): plots bias-variance trade-off for ridge regression
"""
from sklearn.linear_model import Ridge

# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
K = 20
bootstraps = 100
plot_only_best_lambda = True
lambdas = np.logspace(-10, 0, 4)

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
    lambdas = np.logspace(-8, 3, 15)
    lambdas[0] = 0
    ridge = Ridge(fit_intercept=False)
    lam, best_MSE, best_poly = find_best_lambda(X, z, ridge, lambdas, N, K)
    lambdas = [lam]
    f"Optimal lambda = {lam}, best MSE = {best_MSE}, best polynomial = {best_poly}"

raise ValueError
# for lambdas
for i in range(len(lambdas)):
    if not plot_only_best_lambda:
        plt.subplot(411 + i)
        plt.suptitle(f"Bias variance tradeoff for ridge regression")

    # results
    errors = np.zeros(N+1)
    biases = np.zeros(N+1)
    variances = np.zeros(N+1)

    # for polynomial degree
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
            model=ridge,
            lam=lambdas[i],
        )

        # bias-variance trade-off
        error, bias, variance = bias_variance(z_test, z_preds)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    # plot
    plt.plot(biases, label="bias")
    plt.plot(errors, "r--", label="MSE test")
    plt.plot(variances, label="variance")
    plt.xlabel("Polynomial degree (N)", size=12)
    plt.tight_layout(h_pad=0.001)
    if plot_only_best_lambda:
        print(
            f"Optimal lambda = {lam}, best MSE = {best_MSE}, best polynomial = {best_poly}"
        )
        plt.title(
            f"Bias variance tradeoff for ridge regression for optimal lambda = {lambdas[i]}",
            size=18,
        )
    else:
        plt.title(f"lambda = {lambdas[i]:.5}", size=15)
    plt.legend(prop={"size": 10}, loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()
