"""
task b: plots 3D surface plot of Franke Function next to our approximated polynomial fit of the Franke Function, beta
        progression over polynomial degree, and MSE and R2 scores of our implemented OLS model compared with the scikit
        OLS model. The parameter betas_to_plot changes how many betas are plotted, N how many polynomial degrees we test
        for, and scaling for if we center our data or not. The noise parameter controls how much noise we add to our
        Franke Function output. Note that these plots are without resampling as specified in task b.
"""
import matplotlib.pyplot as plt

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
betas_to_plot = 9
noise = 0.05
N = 20
scaling = False

# add noise
z += noise * np.random.standard_normal(z.shape)
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

# model under testing
OLS_model = OLS

# perform linear regression
betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_model
)

# calculate OLS scores without resampling
MSE_train, R2_train = scores(z_train, z_preds_train)
MSE_test, R2_test = scores(z_test, z_preds_test)

# scikitLearn OLS for comparison
OLS_scikit = LinearRegression(
    fit_intercept=False
)  # false because we use our own scaling

# perform linear regression scikit
_, z_preds_train_sk, z_preds_test_sk, _ = linreg_to_N(
    X, X_train, X_test, z_train, z_test, N, scaling=scaling, model=OLS_scikit
)

# calculate OLS scikit scores without resampling
MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

# ------------ PLOTTING 3D -----------------------
fig = plt.figure(figsize=plt.figaspect(0.3))

# Subplot for Franke Function
ax = fig.add_subplot(121, projection="3d")
# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title("Franke Function")
# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)


# Subplot for the prediction
# Plot the surface.
ax = fig.add_subplot(122, projection="3d")
# Plot the surface.
surf = ax.plot_surface(
    x,
    y,
    np.reshape(z_preds[:, np.argmin(MSE_test)], z.shape),
    cmap=cm.coolwarm,
    linewidth=0,
    antialiased=False,
)
# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
ax.set_title(f"Best polynomial fit of Franke Function, N={np.argmin(MSE_test)}")
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# ---------------- PLOTTING GRAPHS --------------
plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
if betas_to_plot <= betas.shape[0]:
    for beta in range(betas_to_plot):
        data = betas[beta, :]
        data[data==0] = np.nan
        plt.plot(data, label=f"beta{beta}", marker="o", markersize=3)
        plt.xlabel("Polynomial degree (N)")
        plt.ylabel("Beta value")
        plt.title("Beta progression")
        plt.legend()
    plt.show()


print(f"Minimal MSE_test value = {np.min(MSE_test)} for N = {np.argmin(MSE_test)}")
plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
plt.plot(MSE_train, label="train implementation", marker="o", markersize=3)
plt.plot(MSE_test, label="test implementation", marker="o", markersize=3)
plt.plot(MSE_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
plt.plot(MSE_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
plt.ylabel("MSE score")
plt.xlabel("Polynomial degree (N)")
plt.title("MSE scores over model complexity")
plt.ylim(0, 0.1)
plt.legend()
plt.show()

plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
plt.plot(R2_train, label="train implementation", marker="o", markersize=3)
plt.plot(R2_test, label="test implementation", marker="o", markersize=3)
plt.plot(R2_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
plt.plot(R2_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
plt.ylabel("R2 score")
plt.xlabel("Polynomial degree (N)")
plt.ylim(-2, 1)
plt.title("R2 scores over model complexity")
plt.legend()
plt.show()