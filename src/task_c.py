# Our own library of functions
from utils import *

np.random.seed(42069)

# make data
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
zs = SkrankeFunction(x, y)

# parameters
N = 20
noise = 0.05
scaling = False
bootstraps = 100

# add noise
z += noise * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

# results
errors = np.zeros(N)
biases = np.zeros(N)
variances = np.zeros(N)

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
    )

    error, bias, variance = bias_variance(z_test, z_preds_test)
    errors[n] = error
    biases[n] = bias
    variances[n] = variance

# plot
print(f"Minimal MSE_test value = {np.min(errors)} for N = {np.argmin(errors)}")
plt.plot(errors, label="error")
plt.plot(biases, label="biases")
plt.plot(variances, label="variances")
plt.xlabel("Polynomial degree")
plt.title("Bias-variance tradeoff over model complexity")
plt.legend()
plt.show()
