"""
task g: performs bias-variance tradeoff for lasso regression using the scikit learn
        lasso model
"""
from sklearn.linear_model import Lasso
# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
N = 20
K = 20
bootstraps = 100
plot_only_best_lambda = False
lambdas = np.logspace(-12, -4, 6)
# synthetic parameters
noise = 0.05
scaling = False

# read in and get data
X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(N, scaling, noise)
z = z.ravel()

# plot only the gridsearched lambda
if plot_only_best_lambda:
    lasso = Lasso(fit_intercept=False)
    lam, _, _ = find_best_lambda(X, z, lasso, lambdas, N, K)
    lambdas = [lam]

# loop through different lambda values
for i in range(len(lambdas)):
    if not plot_only_best_lambda:
           plt.subplot(321 + i)
           plt.suptitle(f"Bias variance tradeoff for lasso regression")

    # model under testing
    model_Lasso = Lasso(lambdas[i], max_iter=200, fit_intercept=False)

    # arrays for bias-variance
    errors = np.zeros(N)
    biases = np.zeros(N)
    variances = np.zeros(N)

    # for polynomial degree
    for n in range(N):
        print(n)
        l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta
        z_preds_test = bootstrap(
            X[:, :l],
            X_train[:, :l],
            X_test[:, :l],
            z_train,
            z_test,
            bootstraps,
            scaling=scaling,
            model=model_Lasso,
            lam=lambdas[i],
        )

        # calculate bias-variance
        error, bias, variance = bias_variance(z_test, z_preds_test)
        errors[n] = error
        biases[n] = bias
        variances[n] = variance

    # plot subplots
    plt.plot(errors, "g--", label="MSE test")
    plt.plot(biases, label="bias")
    plt.plot(variances, label="variance")
    plt.xlabel("Polynomial degree (N)")
    plt.tight_layout(h_pad=0.001)
    if plot_only_best_lambda:
        plt.title(f"Bias variance tradeoff for lasso regression for optimal lambda = {lambdas[i]}")
    else:
        plt.title(f"lambda = {lambdas[i]:.5}")
    plt.legend()


plt.show()