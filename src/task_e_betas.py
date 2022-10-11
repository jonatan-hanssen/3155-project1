"""
task g: calculate and plot ridge beta value over model complexity
"""
# Our own library of functions
from utils import *

np.random.seed(42069)

# Parameters
lambdas = np.logspace(-10, 0, 4)
betas_to_plot = 9

# Parameters for synthetic data

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

# calculate beta values
for i in range(len(lambdas)):
    plt.subplot(411 + i)
    plt.suptitle(f"Beta progression for different lambdas")
    betas, z_preds_train, z_preds_test, _ = linreg_to_N(
        X,
        X_train,
        X_test,
        z_train,
        z_test,
        N,
        centering=centering,
        model=ridge,
        lam=lambdas[i],
    )

    # plot beta values
    if betas_to_plot <= betas.shape[0]:
        for beta in range(betas_to_plot):
            data = betas[beta, :]
            data[data == 0] = np.nan
            plt.plot(data, label=f"beta{beta}", marker="o", markersize=3)
    plt.xlabel("Polynomial degree (N)", size=15)
    plt.ylabel("Beta value", size=15)
    plt.title("Beta progression", size=18)
    plt.title(f"lambda = {lambdas[i]:.5}")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

plt.show()
