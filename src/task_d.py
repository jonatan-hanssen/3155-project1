import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold

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
K = 10
bootstraps = 100

# add noise
z += noise * np.random.standard_normal(z.shape)

# split datasets
X, X_train, X_test, z_train, z_test = preprocess(x, y, z, N, 0.2)

z = z.ravel()

# model under testing
OLS_model = OLS

# scikit model under testing
OLS_model_scikit = LinearRegression(fit_intercept=scaling)
kfolds = KFold(n_splits=K)

# results cross validation
errors_cv = np.zeros(N)
errors_cv_scikit = np.zeros(N)

# results bootstrap
errors_boot = np.zeros(N)
biases_boot = np.zeros(N)
variances_boot = np.zeros(N)

# cross validation
print("Cross")
for n in range(N):
    print(n) # progress counter
    l = int((n + 1) * (n + 2) / 2)  # Number of elements in beta

    # own implementation
    errors_cv[n] = crossval(X[:, :l], z, K, scaling=scaling, model=OLS_model)

    # scikit
    error_scikit = cross_validate(
        estimator=OLS_model_scikit,
        X=X[:, :l],
        y=z,
        scoring="neg_mean_squared_error",
        cv=kfolds,
    )
    errors_cv_scikit[n] = np.mean(-error_scikit["test_score"])

# bootstrap
print("Boot")
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
        model=OLS_model,
    )

    error, bias, variance = bias_variance(z_test, z_preds_test)
    errors_boot[n] = error
    biases_boot[n] = bias
    variances_boot[n] = variance

plt.plot(errors_boot, label="bootstrap")
plt.plot(errors_cv, label="cross validation implementation")
plt.plot(errors_cv_scikit, label="cross validation scikit learn")
plt.xlabel("Polynomial Degree (N)")
plt.ylabel("MSE score")
plt.suptitle(
    "MSE by resampling method")
plt.title(f"Datapoints = {len(x)*len(y)}, Parameters: N = {N}, noise = {noise}, scaling = {scaling}, K = {K}, bootstraps = {bootstraps}, scaling={scaling}, n={N}, k={K}, bootstraps={bootstraps}", fontsize=6)
plt.ylim(0, 0.1)

plt.legend()
plt.show()
