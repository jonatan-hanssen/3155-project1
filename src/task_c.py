"""
task c: bootstrapped bias-variance trade-off plot for OLS on synthetic data
"""
# Our own library of functions
from utils import *
import matplotlib

np.random.seed(42069)

# make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
zs = SkrankeFunction(x, y)

# parameters
N = 25
noise = 0.05
scaling = True
bootstraps = 100

# model under testing
model = OLS

# add noise
# z += noise * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

# results
MSEs_test = np.zeros(N)
MSEs_train = np.zeros(N)
MSEs_test_sk = np.zeros(N)
MSEs_train_sk = np.zeros(N)

biases = np.zeros(N)
variances = np.zeros(N)

ols_scikit = LinearRegression(fit_intercept=False)
# bootstrap
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
        model=model,
    )

    z_preds_test_sk = bootstrap(
        X[:, :l],
        X_train[:, :l],
        X_test[:, :l],
        z_train,
        z_test,
        bootstraps,
        scaling=scaling,
        model=ols_scikit,
    )

    _, z_pred_train, _, _ = evaluate_model(
        X[:, :l], X_train[:, :l], X_test[:, :l], z_train, model
    )

    _, z_pred_train_sk, _, _ = evaluate_model(
        X[:, :l], X_train[:, :l], X_test[:, :l], z_train, ols_scikit
    )

    MSE_test, bias, variance = bias_variance(z_test, z_preds_test)
    MSE_train = MSE(z_pred_train, z_train)

    MSE_test_sk, _, _ = bias_variance(z_test, z_preds_test_sk)
    MSE_train_sk = MSE(z_pred_train_sk, z_train)

    MSEs_test[n] = MSE_test
    MSEs_test_sk[n] = MSE_test_sk

    MSEs_train[n] = MSE_train
    MSEs_train_sk[n] = MSE_train_sk

    biases[n] = bias
    variances[n] = variance

# plot

font = {"family": "normal", "size": 25}
matplotlib.rc("font", **font)

print(f"Minimal MSE_test value = {np.min(MSEs_test)} for N = {np.argmin(MSEs_test)}")
plt.plot(MSEs_test, label="error", marker="o", markersize=3, linewidth=4)
plt.plot(biases, label="biases", marker="o", markersize=3, linewidth=4)
plt.plot(variances, label="variances", marker="o", markersize=3, linewidth=4)
plt.xlabel("Polynomial degree (N)")
plt.legend()
plt.show()

plt.plot(
    MSEs_train, label="train own implementation", marker="o", markersize=4, linewidth=3
)
plt.plot(
    MSEs_test, label="test own implementation", marker="o", markersize=4, linewidth=3
)
plt.plot(
    MSEs_train_sk,
    "r--",
    label="train ScikitLearn",
    marker="o",
    markersize=4,
    linewidth=3,
)
plt.plot(
    MSEs_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=4, linewidth=3
)
# plt.ylim(0, 0.1)
plt.ylabel("MSE score")
plt.xlabel("Polynomial degree (N)")
plt.title("MSE scores over model complexity")
plt.legend()
plt.show()
