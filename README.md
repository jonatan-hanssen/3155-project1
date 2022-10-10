# Project 1 for FYS-STK3155/4155

This is the repo for the first assignment in FYS-STK3155. All code pair-programmed by Jonatan Hanssen and Eric Reber. Source code is located in ```src```. The scientific report is contained in the file ```doc/paper.pdf```.



## Run instructions
To run a program for synthetic data (the franke function) and all default values, simply run the program, for example:
```
python3 task_b.py
```

To run a program for real data (for example the SRTM data provided in ../data/), run the program with the filename as a parameter, for example:
```
python3 task_b.py -f ../data/nearest_neighbor_SRTM_data_Norway_1.tif
```

If you would like to change other parameters, the following optional arguments are provided from the command line

```
usage: task_*.py [-h] [-f FILE | -d | -no NOISE] [-st STEP] [-b BETAS] [-n N]
                 [-sc]

Read in arguments for tasks

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  Terrain data file name
  -d, --debug           Use debug function for testing. Default false
  -no NOISE, --noise NOISE
                        Amount of noise to have. Recommended range [0-0.1].
                        Default 0.05
  -st STEP, --step STEP
                        Step size for linspace function. Range [0.01-0.4].
                        Default 0.05
  -b BETAS, --betas BETAS
                        Betas to plot, when applicable. Default 10
  -n N                  Polynomial degree. Default 10
  -sc, --scaling        Whether to use scaling (centering for synthetic case
                        or MinMaxScaling for organic case)
```

If you wish to change even more parameters (at your own risk), the following advanced parameters are provided at the top of each script:

K = number of folds in K-folds : int \
bootstraps = number of bootstraps : int \
lambdas = array of lambdas to test : ndarray \
plot_only_best_lambda = plots only lambda which gives least MSE : bool
