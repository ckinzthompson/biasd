''' This script loads the example data, sets some priors, and then finds the Laplace approximation to the posterior distribution.'''

## Imports
import matplotlib.pyplot as plt
import numpy as np
import biasd as b


#### Setup the analysis
## Load the SMD example dataset
filename = './example_dataset.hdf5'
dataset = b.smd.load(filename)

## Select the data from the first trajectory
trace = dataset['trajectory 0']
time = trace['data/time'].value
fret = trace['data/E_{FRET}'].value

## Parse meta-data to load time resolution
tau = trace['data'].attrs['tau']

## Get the simulation ground truth values 
truth = trace['data/simulation'].attrs['truth']

## Close the dataset
dataset.close()


#### Perform a Calculation
## Make the prior distribution
## set means to ground truths: (.1, .9, .05, 3., 8.)
e1 = b.distributions.normal(0.1, 0.2)
e2 = b.distributions.normal(0.9, 0.2)
sigma = b.distributions.gamma(1., 1./0.05)
k1 = b.distributions.gamma(1., 1./3.)
k2 = b.distributions.gamma(1., 1./8.)
priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

## Find the Laplace approximation to the posterior 
posterior = b.laplace.laplace_approximation(fret,priors,tau)

## Calculate the predictive posterior distribution for visualization
x = np.linspace(-.2,1.2,1000)
samples = posterior.samples(100)
predictive = b.likelihood.predictive_from_samples(x,samples,tau)


#### Save this analysis
## Load the dataset file
dataset = b.smd.load(filename)

## Create a new group to hold the analysis in 'trajectory 0'
trace = dataset['trajectory 0']
laplace_analysis = trace.create_group("Laplace analysis 20161230")

## Add the priors
b.smd.add.parameter_collection(laplace_analysis,priors,label='priors')

## Add the posterior
b.smd.add.laplace_posterior(laplace_analysis,posterior,label='posterior')

## Add the predictive
laplace_analysis.create_dataset('predictive x',data = x)
laplace_analysis.create_dataset('predictive y',data = predictive)

## Save and close the dataset
b.smd.save(dataset)


#### Visualize the results
## Plot a histogram of the data
plt.hist(fret, bins=71, range=(-.2,1.2), normed=True, histtype='stepfilled', alpha=.6, color='blue', label='Data')

## Plot the predictive posterior of the Laplace approximation solution
plt.plot(x, predictive, 'k', lw=2, label='Laplace')

## Uncomment below to plot the predictive of the  for comparison. Note that the accuracy of this calculation depends on sufficient sampling, and since the priors are broad, it is difficult to sample well enough to get a smooth distribution.
# samples = priors.rvs(500).T
# predictive_prior = b.likelihood.predictive_from_samples(x,samples,tau)
# plt.plot(x, predictive_prior, 'g', lw=2, label='Priors')

## We know the data was simulated, so:
## plot the probability distribution used to simulate the data
plt.plot(x, np.exp(b.likelihood.nosum_log_likelihood(truth, x, tau)), 'r', lw=2, label='Truth')

## Label Axes and Curves
plt.ylabel('Probability',fontsize=18)
plt.xlabel('Signal',fontsize=18)
plt.legend()

## Make the Axes Pretty
a = plt.gca()
a.spines['right'].set_visible(False)
a.spines['top'].set_visible(False)
a.yaxis.set_ticks_position('left')
a.xaxis.set_ticks_position('bottom')

# Save the figure, then show it
plt.savefig('example_laplace_predictive.png')
plt.show()