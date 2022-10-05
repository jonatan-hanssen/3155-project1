# Our own library of functions
from utils import *

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)
# z = SkrankeFunction(x, y)
N = 12
bootstraps = 100

np.random.seed(42069)

z += 0.05 * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

errors = np.zeros(N)
biases = np.zeros(N)
variances = np.zeros(N)

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
        scaling=True,
    )

    error, bias, variance = bias_variance(z_test, z_preds)
    errors[n] = error
    biases[n] = bias
    variances[n] = variance

print(f"Minimal MSE_test value = {np.min(errors)} for N = {np.argmin(errors)}")
plt.plot(errors, label="error")
plt.plot(biases, label="biases")
plt.plot(variances, label="variances")
plt.xlabel("Polynomial degree")
plt.title("Bias-variance tradeoff over model complexity")
plt.legend()
plt.show()
