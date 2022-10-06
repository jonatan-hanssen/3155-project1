"""
task b (and task g): plot terrain, approximate terrain with OLS and calculate MSE, R2 beta over model complexity for
                     real data. Performs task_b, so no resampling.
"""
from imageio import imread
import sys
import argparse
import matplotlib

# Our own library of functions
from utils import *

np.random.seed(42069)

argv = sys.argv[1:]

parser = argparse.ArgumentParser(description="Compute task g.b")

# filename is optional
parser.add_argument("-f", "--file", help="The filename to apply filter to")

# parse arguments and call run_filter
args = parser.parse_args()

# parameters
N = 25
betas_to_plot = 9
noise = 0.05
scaling = False

# implemented model under testing
OLS_model = OLS
# scikit model under testing
OLS_scikit = LinearRegression(fit_intercept=False)

if args.file:
    # Load the terrain
    z = np.asarray(imread(args.file), dtype="float64")
    x = np.arange(z.shape[0])
    y = np.arange(z.shape[1])
    x, y = np.meshgrid(x, y, indexing="ij")

    # split data into test and train
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # normalize data
    X, X_train, X_test, z, z_train, z_test = minmax_dataset(
        X, X_train, X_test, z, z_train, z_test
    )

    # perform linear regression
    betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, model=OLS_model
    )

    # perform linear regression scikit
    _, z_preds_train_sk, z_preds_test_sk, z_preds_sk = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, model=OLS_scikit
    )

else:
    x = np.arange(0, 1, 0.05)
    y = np.arange(0, 1, 0.05)
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)

    # add noise
    z += noise * np.random.standard_normal(z.shape)
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # perform linear regression
    betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_model
    )

    # perform linear regression scikit
    _, z_preds_train_sk, z_preds_test_sk, _ = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_scikit
    )

# Calculate OLS scores
MSE_train, R2_train = scores(z_train, z_preds_train)
MSE_test, R2_test = scores(z_test, z_preds_test)

# calculate OLS scikit scores without resampling
MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

# approximation of terrain (2D plot)
pred_map = z_preds[:, -1].reshape(z.shape)

# ------------ PLOTTING 3D -----------------------

font = {"family": "normal", "size": 25}

matplotlib.rc("font", **font)

fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for terrain
ax = fig.add_subplot(121, projection="3d")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Scaled terrain")
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

# Subplot for the prediction
# Plot the surface.
ax = fig.add_subplot(122, projection="3d")
# Plot the surface.
surf = ax.plot_surface(
    x,
    y,
    np.reshape(z_preds[:, N], z.shape),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Polynomial fit of scaled terrain")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# ---------------- PLOTTING GRAPHS --------------
plt.subplot(121)
plt.title("Terrain 1")
plt.imshow(z)
plt.colorbar()
plt.subplot(122)
plt.title("Predicted terrain 1")
plt.imshow(pred_map)
plt.colorbar()
plt.show()

#    plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
if betas_to_plot <= betas.shape[0]:
    for beta in range(betas_to_plot):
        data = betas[beta, :]
        data[data == 0] = np.nan
        plt.plot(data, label=f"beta{beta}", marker="o", markersize=3)
        plt.xlabel("Polynomial degree (N)")
        plt.ylabel("Beta value")
        plt.title("Beta progression")
        plt.legend()
    plt.show()

print(f"Minimal MSE_test value = {np.min(MSE_test)} for N = {np.argmin(MSE_test)}")
#    plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
plt.plot(
    MSE_train, label="train own implementation", marker="o", markersize=4, linewidth=3
)
plt.plot(
    MSE_test, label="test own implementation", marker="o", markersize=4, linewidth=3
)
plt.plot(
    MSE_train_sk,
    "r--",
    label="train ScikitLearn",
    marker="o",
    markersize=4,
    linewidth=3,
)
plt.plot(
    MSE_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=4, linewidth=3
)
plt.ylim(0, 0.1)
plt.ylabel("MSE score")
plt.xlabel("Polynomial degree (N)")
plt.title("MSE scores over model complexity")
plt.legend()
plt.show()

# plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
plt.plot(R2_train, label="train implementation", marker="o", markersize=4, linewidth=3)
plt.plot(R2_test, label="test implementation", marker="o", markersize=4, linewidth=3)
plt.plot(
    R2_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=4, linewidth=3
)
plt.plot(
    R2_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=4, linewidth=3
)
plt.ylim(-2, 1)
plt.ylabel("R2 score")
plt.xlabel("Polynomial degree (N)")
plt.title("R2 scores over model complexity")
plt.legend()
plt.show()
