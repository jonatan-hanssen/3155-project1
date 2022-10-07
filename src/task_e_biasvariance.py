"""
task e (and task g): plots bias-variance trade-off for ridge regression
"""
from sklearn.linear_model import Ridge
# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
K = 20
N = 10
bootstraps = 100
plot_only_best_lambda = True
lambdas = np.logspace(-10, 0, 6)
# parameter synthetic data
noise = 0.05
scaling = False

# read in data
X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(
    N, scaling, noise
)
z = z.ravel()

# plot only the gridsearched lambda
if plot_only_best_lambda:
    ridge = Ridge(fit_intercept=False)
    lam, _, _ = find_best_lambda(X, z, ridge, lambdas, N, K)
    lambdas = [lam]

# for lambdas
for i in range(len(lambdas)):
    if not plot_only_best_lambda:
       plt.subplot(321 + i)
       plt.suptitle(f"Bias variance tradeoff for ridge regression")

    # results
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
            scaling=scaling,
            model=ridge,
            lam=lambdas[i],
        )

        # bias-variance trade-off
        error, bias, variance = bias_variance(z_test, z_preds)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    # plot
    plt.plot(errors, label="MSE test")
    plt.plot(biases, label="bias")
    plt.plot(variances, label="variance")
    plt.xlabel("Polynomial degree (N)")
    plt.tight_layout(h_pad=0.001)
    if plot_only_best_lambda:
        plt.title(f"Bias variance tradeoff for ridge regression for optimal lambda = {lambdas[i]}")
    else:
        plt.title(f"lambda = {lambdas[i]:.5}")
    plt.legend()

plt.show()