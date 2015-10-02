# Running Bayesian Inference for the Analysis of Sub-temporal-resolution Data (BIASD)
## Requirements
You must have Python as well as several packages (Numpy, Matplotlib, and Scipy) to run BIASD. These can all be obtained easily by installing a Python distribution such as Anaconda from <https://www.continuum.io/downloads>, which is free. Note that the GUI does not work on Windows, and looks bad on Mac. The best option is to use Linux.

## Scripting BIASD
The file biasd.py contains a series of functions and classes that allow BIASD to be easily run. This allows multiple dataset to be processed by scripting a new python file. An example script is this:

```python
from biasd import *

mypriors = biasddistribution(
dist('normal',0.1,.1),
dist('normal',.9,.1),
dist('gamma',50.,1000.),
dist('Gamma',2.,2./10.),
dist('Gamma',2.,2./15.))

d = dataset(data_fname = 'testdataset.dat', fmt='1D', tau=0.1, priors=mypriors, analysis_fname = 'testdataset.biasd')

d.load_data()
d.run_laplace(nproc=8)
d.variational_ensemble(nstates=10,nproc=8)
d.save_analysis()
```

After running, the dataset and the results of the Laplace approximation and the variational Gaussian mixture model will be saved into the file 'testdataset.biasd'. This file can easily be loaded for subsequent analysis.

```python
from biasd import *

d = dataset(analysis_fname = 'testdataset.biasd')
d.load_analysis()

#Example analysis
for trace in d.traces:
	print trace.posterior.mu
```

## BIASD Graphical User Interface (GUI)
The file gui_biasd.py contains a GUI that uses the functions in the file biasd.py to perform the analysis of a dataset that is setup through the GUI. There are four tabs in the GUI: Main, Trajectories, Priors, Analysis.

### Main tab
Signal versus time trajectories can be loaded in by clicking the *Load Data* button. The data should be in a tab delimited file in one of three formats:

1) 2D NxT, where each row of the file is one of the N signal versus time trajectories, and each column is a new time period. Trajectories of disparate lengths can be entered by using NaN of infinity in time periods of trajectories where there is no signal.
2) 2D TxN, where the format is similar to 2D NxT but each row corresponds to a new time period, and each column is a different trajectory.
3) 1D, where the first row of the file is a label vector, and the second row is a data vector. The signal values from the trajectories are concatenated to form the data vector. A unique identifying number for each trajectory is located in the same column as the data points from a particular trajectory, but in the label vector. 

The time period of the measurements in the signal versus time trajectories that were loaded in is set by the *Set Tau* button. The GUI can be reset by the *New Analysis* button. The *Save Analysis* button will save the parameters (e.g., tau, priors), and any results (e.g., Laplace approximations, Ensemble analysis) from an analysis into a file. These files can be loaded back using the *Load Analysis* button. Note that after an analysis action, the results are not saved, so the *Save Analysis* must be used before exiting.

The *Load Integrand* button is used to point to the location of the compiled library containing the BIASD likelihood function integrand. This is compiled from the C-file integrand_gsl.c. It is highly advised to use this library instead of the native python-version of the integrand, because that is much slower (~30x). The speed with which a single datapoint can be analyzed using BIASD can be tested using the *Test Speed* button.

### Trajectories Tab
If any data has been loaded in, you can preview the signal versus time trajectories using this tab. Use this function to ensure that the data format on the Main tab is correct. Each trajectory is shown in the upper plot, and a histogram of the entire dataset is shown in the lower plot. Trajectories can be switched using the buttons at the bottom of the tab (Start, -10, -1, +1, +10, End), or the arrow keys. The left arrow decreases the trajectory number by 1, and the right arrow increases the trajectory number by 1. Holding the shift key changes jump to +/- 10 trajectories.

### Priors Tab
The priors for a subsequent analysis are set from this tab. The parameter for each distribution is shown on the left. The distribution type for that parameter may be changed using the drop-down box to uniform, normal, gamma, or beta. Parameters 1 and 2 correspond to the parameters of that distribution type. Uniform - p1 = a, p2 = b; Normal - p1 = mu, p2 = sigma; Gamma - p1 = alpha, p2 = beta; Beta - p1 = alpha, p2 = beta.
After changing any of these values or the distribution type, you must click the *Check & Set Priors* button. Plot priors will then display the probability distribution functions of the priors, as well as an approximate marginalized probability of the data.

### Analysis Tab
After setting up the dataset using the other tabs, BIASD is run on the analysis tab. Set the number of CPUs you'd like to use for the calculations in the *# CPU* drop-down menu. Set the Maximum number of states to use for the ensemble button in the *Max Stats* drop-down menu. Clicking the *Run* button will find the Laplace approximation of the posteriors for each signal versus time trajectory. You must click the *Save Analysis* button on the Main tab afterwards to save these results. Following this, you may use a variational Gaussian mixture model to calculate the ensemble properties of the dataset by using the *Ensemble* button. Once this step has completed, you may save the displayed plot and text by pushing the *S* key. You may also change the number of states displayed by pressing control and then the number of states to display. Remember to click the *Save Analysis* button to save the ensemble results.

## Compiling the Integrand Library
The C-based integrand is provide in the file integrand_gsl.c. It must be compiled to be used. This programs evaluates Bessel functions using functions from the GNU Scientific Library (GSL), though they have been embedded in the file so that you do not have to install GSL. To compile the integrand library try:

```bash
gcc -L/usr/local/lib -shared -o integrand_gsl.so -fPIC -O3 integrand_gsl.c -lm
```