"""
task c (and task g): plots bias-variance tradeoff with bootstrap for OLS
"""
import matplotlib.pyplot as plt

# Our own library of functions
from utils import *

np.random.seed(42069)

# parameters
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

# result arrays
MSEs_test = np.zeros(N)
MSEs_train = np.zeros(N)
biases = np.zeros(N)
variances = np.zeros(N)

# scikit
MSEs_test_sk = np.zeros(N)
MSEs_train_sk = np.zeros(N)


# model under testing
OLS_model = OLS
OLS_scikit = LinearRegression()

# for polynomial degree
for n in range(N):
    print(n)
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

    # bootstrap
    z_preds_test, z_pred_train = bootstrap(
        X[:, :l],
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        centering=centering,
        model=OLS_model,
    )
    MSE_test, bias, variance = bias_variance(z_test, z_preds_test)
    MSE_train = MSE(z_pred_train, z_train)

    MSEs_test[n] = MSE_test
    MSEs_train[n] = MSE_train

    biases[n] = bias
    variances[n] = variance

    z_preds_test_sk, z_pred_train_sk = bootstrap(
        X[:, :l],
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        centering=centering,
        model=OLS_scikit,
    )

    mse_scores_sk, _ = scores(z_test, z_preds_test_sk)
    MSE_test_sk = np.mean(mse_scores_sk)
    MSE_train_sk = MSE(z_pred_train_sk, z_train)

    MSEs_test_sk[n] = MSE_test_sk
    MSEs_train_sk[n] = MSE_train_sk

print(f"Minimal MSE_test value = {np.min(MSEs_test)} for N = {np.argmin(MSEs_test)}")
# plot mse test and train
plt.plot(MSEs_train, label="train implementation", marker="o", markersize=3)
plt.plot(MSEs_test, label="test implementation", marker="o", markersize=3)
plt.plot(MSEs_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
plt.plot(MSEs_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
plt.xlabel("Polynomial degree (N)", size=15)
plt.ylabel("MSE score", size=15)
plt.title("MSE scores over model complexity", size=18)
plt.legend(prop={"size": 12})
plt.show()

# plot
plt.plot(biases, label="bias")
plt.plot(MSEs_test, "r--", label="MSE test")
plt.plot(variances, label="variances")
plt.xlabel("Polynomial degree (N)", size=12)
plt.title("Bias-variance tradeoff over model complexity OLS", size=15)
plt.legend(prop={"size": 12})
plt.show()
