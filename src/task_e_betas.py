"""
task g: calculate and plot ridge beta value over model complexity
"""
# Our own library of functions
from utils import *

np.random.seed(42069)

# Parameters
N = 20
lambdas = np.logspace(-15, -9, 6)
lambdas[-1] = 10000000000
betas_to_plot = 9

# Parameters for synthetic data
noise = 0.05
scaling = False

# get data
X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(
    N, scaling, noise
)

# calculate beta values
for i in range(len(lambdas)):
    plt.subplot(321 + i)
    betas, z_preds_train, z_preds_test, _ = linreg_to_N(
        X,
        X_train,
        X_test,
        z_train,
        z_test,
        N,
        scaling=scaling,
        model=ridge,
        lam=lambdas[i],
    )

    # plot beta values
    if betas_to_plot <= betas.shape[0]:
        for col in range(betas_to_plot):
            plt.plot(betas[col, :], label=f"beta{col}", marker="o", markersize=3)

    plt.title(f"Beta progression for lambda = {lambdas[i]:.5}")
    plt.legend()

plt.show()
