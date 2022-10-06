"""
USELESS, PLEASE DELETE
"""
"""
task f: evaluates lasso and ridge to find the lambda value yielding the best MSE score.
"""
from sklearn.linear_model import Ridge, Lasso

# Our own library of functions
from utils import *

np.random.seed(42069)
# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y)

# Highest order polynomial we fit with
N = 20
K = 20
noise = 0.05
scaling = True

X, X_train, X_test, z, z_train, z_test, scaling, x, y, z = read_in_dataset(N, scaling, noise)
z = z.ravel()

lambdas = np.logspace(-12, 0, 25)
lambdas[-1] = 0
ridge = Ridge(fit_intercept=scaling)
lasso = Lasso(fit_intercept=scaling)

find_best_lambda(X, z, ridge, lambdas, N, K)
#find_best_lambda(X, z, lasso, lambdas, N, K)