"""
task c (and task g): plots bias-variance tradeoff with bootstrap for OLS
"""
# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
N = 20
bootstraps = 100
# parameters for synthetic data
noise = 0.05
scaling = False

# get data
X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(N, scaling, noise)

# result arrays
errors = np.zeros(N)
biases = np.zeros(N)
variances = np.zeros(N)

# model under testing
OLS_model = OLS

# for polynomial degree
for n in range(N):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

    # bootstrap
    z_preds_test = bootstrap(
        X[:, :l],
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        scaling=scaling,
        model=OLS_model,
    )

    # bias-variance trade-off
    error, bias, variance = bias_variance(z_test, z_preds_test)
    errors[n] = error
    biases[n] = bias
    variances[n] = variance

# plot
plt.plot(errors, 'g--', label="MSE test")
plt.plot(biases, label="biases")
plt.plot(variances, label="variances")
plt.xlabel("Polynomial degree (N)")
plt.title("Bias-variance tradeoff over model complexity OLS")
plt.legend()
plt.show()