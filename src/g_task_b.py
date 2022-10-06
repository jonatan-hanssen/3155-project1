"""
task g: plot terrain, approximate terrain with OLS and calculate MSE, R2
        beta over model complexity for real data. Performs task_b, so no
        resampling.
"""
from imageio import imread

# Our own library of functions
from utils import *

np.random.seed(42069)

# Load the terrain
z_terrain1 = imread("../data/small_SRTM_data_Norway_1.tif")
x_terrain1 = np.arange(z_terrain1.shape[0])
y_terrain1 = np.arange(z_terrain1.shape[1])
x1, y1 = np.meshgrid(x_terrain1, y_terrain1, indexing="ij")

z_terrain2 = imread("../data/SRTM_data_Norway_2.tif")
x_terrain2 = np.arange(z_terrain2.shape[0])
y_terrain2 = np.arange(z_terrain2.shape[1])
x2, y2 = np.meshgrid(x_terrain2, y_terrain2, indexing="ij")

# Show the terrain
plt.plot()
plt.subplot(121)
plt.title("Terrain over Norway 1")
plt.imshow(z_terrain1)
plt.xlabel("Y")
plt.ylabel("X")

plt.subplot(122)
plt.title("Terrain over Norway 2")
plt.imshow(z_terrain2)
plt.xlabel("Y")
plt.ylabel("X")
plt.show()

# Highest order polynomial we fit with
N = 5
betas_to_plot = 9

def task_b(x, y, z, N):
    # split data into test and train
    X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

    # normalize data
    X, X_train, X_test, z, z_train, z_test = normalize_task_g(
        X, X_train, X_test, z, z_train, z_test
    )

    # implemented model under testing
    OLS_model = OLS

    # perform linear regression
    betas, z_preds_train, z_preds_test, z_preds = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, model=OLS_model
    )

    # approximation of terrain (2D plot)
    pred_map = z_preds[:, -1].reshape(z.shape)

    # Calculate OLS scores
    MSE_train, R2_train = scores(z_train, z_preds_train)
    MSE_test, R2_test = scores(z_test, z_preds_test)

    # scikit model under testing
    OLS_scikit = LinearRegression(fit_intercept=False)
    OLS_scikit.fit(X_train, z_train)

    # perform linear regression
    _, z_preds_train_sk, z_preds_test_sk, z_preds_sk = linreg_to_N(
        X, X_train, X_test, z_train, z_test, N, model=OLS_scikit
    )

    # ScikitLearn OLS scores for comparison
    MSE_train_sk, R2_train_sk = scores(z_train, z_preds_train_sk)
    MSE_test_sk, R2_test_sk = scores(z_test, z_preds_test_sk)

    # ------------ PLOTTING 3D -----------------------
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
            data[data==0] = np.nan
            plt.plot(data, label=f"beta{beta}", marker="o", markersize=3)
            plt.xlabel("Polynomial degree (N)")
            plt.ylabel("Beta value")
            plt.title("Beta progression")
            plt.legend()
        plt.show()


    print(f"Minimal MSE_test value = {np.min(MSE_test)} for N = {np.argmin(MSE_test)}")
#    plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
    plt.plot(MSE_train, label="train implementation", marker="o", markersize=3)
    plt.plot(MSE_test, label="test implementation", marker="o", markersize=3)
    plt.plot(MSE_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
    plt.plot(MSE_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
    plt.ylabel("MSE score")
    plt.xlabel("Polynomial degree (N)")
    plt.title("MSE scores over model complexity")
#    plt.ylim(0, 0.1)
    plt.legend()
    plt.show()

    #plt.suptitle(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}", fontsize=6)
    plt.plot(R2_train, label="train implementation", marker="o", markersize=3)
    plt.plot(R2_test, label="test implementation", marker="o", markersize=3)
    plt.plot(R2_train_sk, "r--", label="train ScikitLearn", marker="o", markersize=3)
    plt.plot(R2_test_sk, "g--", label="test ScikitLearn", marker="o", markersize=3)
    plt.ylabel("R2 score")
    plt.xlabel("Polynomial degree (N)")
#    plt.ylim(-2, 1)
    plt.title("R2 scores over model complexity")
    plt.legend()
    plt.show()

task_b(x2, y2, z_terrain2, N)