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
N = 10
noises = np.linspace(0, 0.2, 5)
scaling = False
bootstraps = 100

def noise_plot(x, y, z, N, noise, scaling, bootstraps):
    # add noise
    z += noise * np.random.standard_normal(z.shape)
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # results
    test_errors = np.zeros(N)

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

        MSE_test, _ = scores(z_test, z_preds_test)
        test_errors[n] = np.mean(MSE_test)

    return test_errors

for noise in noises:
    test_errors = noise_plot(x, y, z, N, noise, scaling, bootstraps)
    # plot
    plt.plot(test_errors, label=f"test error noise={round(noise)}", marker="o", markersize=3)
plt.ylim(0, 0.4)
plt.xlabel("Polynomial degree")
plt.ylabel("MSE score")
plt.title("MSE for different levels of noise in model")
plt.legend()
plt.show()