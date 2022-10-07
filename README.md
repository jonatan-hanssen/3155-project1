# Project 1 for 3155/4155

This is the repo for the first assignment in FYS-STK3155. All code pair-programmed by Jonatan Hanssen and Eric Reber.


# Run instructions
To run a program for synthetic data (the franke function), simply run the program, for example:
python3 task_b.py

To run a program for real data (for example the SRTM data provided in ../data/), run the program with the filename as a parameter, for example:
python3 task_b.py -f ..\data\nearest_neighbor_SRTM_data_Norway_1.tif


# Parameters
N = number of polynomial degrees to run : int
K = number of folds in K-folds : int
bootstraps = number of bootstraps : int
lambdas = array of lambdas to test : ndarray
plot_only_best_lambda = plots only lambda which gives least MSE : bool
betas_to_plot = number of betas plotted on beta progression plot : int

# Additional parameters for synthetic data
scaling = use our own centering implementation (note that real data automatically uses minmax, so we don't use our centering implementation) : bool
noise = how much noise is added to the synthetic data : float
step = amount of datapoints produced, default = 0.05