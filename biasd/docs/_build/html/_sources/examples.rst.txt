.. _examples:

BIASD Examples
==============

Here are some example Python scripts to perform BIASD. They can be found in `./example_data`, along with simulated data in a tab-delimited format (`./example_data/example_data.dat`), and an example HDF5 SMD dataset containing this data and some analysis results (`./example_data/example_dataset.hdf5`).

Creating a new SMD file
-----------------------

This script loads the example data, and then creates an HDF5 SMD data file to contain this data. Future analysis performed with BIASD can also be saved into this file.

.. code-block:: python

	## Imports
	import numpy as np
	import biasd as b

	## Create a new SMD file
	filename = './example_dataset.hdf5'
	dataset = b.smd.new(filename)

	## Load example trajectories (N,T)
	example_data = b.smd.loadtxt('example_data.dat')
	n_molecules, n_datapoints = example_data.shape

	## These signal versus time trajectories were simulated to be like smFRET data.
	## The simulation parameters were:
	tau = 0.1 # seconds
	e1 = 0.1 # E_{FRET}
	e2 = 0.9 # E_{FRET}
	sigma = 0.05 #E_{FRET}
	k1 = 3. # s^{-1}
	k2 = 8. # s^{-1}

	truth = np.array((e1,e2,sigma,k1,k2))

	## Create a vector with the time of each datapoint
	time = np.arange(n_datapoints) * tau

	## Add the trajectories to the SMD file automatically
	b.smd.add.trajectories(dataset, time, example_data, x_label='time', y_label='E_{FRET}')

	## Add some metadata about the simulation to each trajectory
	for i in range(n_molecules):
	
		# Select the group of interest
		trajectory = dataset['trajectory ' + str(i)]
	
		# Add an attribute called tau to the data group.
		# This group contains the time and signal vectors.
		trajectory['data'].attrs['tau'] = tau
	
		# Add a new group called simulation in the data group
		simulation = trajectory['data'].create_group('simulation')
	
		# Add relevant simulation paramters
		simulation.attrs['tau'] = tau
		simulation.attrs['e1'] = e1
		simulation.attrs['e2'] = e2
		simulation.attrs['sigma'] = sigma
		simulation.attrs['k1'] = k1
		simulation.attrs['k2'] = k2
	
		# Add an array of simulation parameters for easy access
		simulation.attrs['truth'] = truth

	## Save the changes, and close the HDF5 file
	b.smd.save(dataset)
	

Sample the posterior with MCMC
------------------------------
This script loads the example data from above, sets some priors, and then uses the Markov chain Monte Carlo (MCMC) technique to sample the posterior.

.. code-block:: python
	
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

	## Setup the MCMC sampler to use 100 walkers and 4 CPUs
	nwalkers = 100
	ncpus = 4
	sampler, initial_positions = b.mcmc.setup(fret, priors, tau, nwalkers, threads = ncpus)

	## Burn-in 100 steps and then remove them form the sampler,
	## but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler,initial_positions,nsteps=100)

	## Run 100 steps starting at the burned-in positions. Timing data will provide an idea of how long each step takes
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=100,timer=True)

	## Continue on from step 100 for another 900 steps. Don't display timing.
	sampler = b.mcmc.continue_run(sampler,900,timer=False)

	## Get uncorrelated samples from the chain by skipping samples according to the autocorrelation time of the variable with the largest autocorrelation time
	uncorrelated_samples = b.mcmc.get_samples(sampler,uncorrelated=True)

	## Make a corner plot of these uncorrelated samples
	fig = b.mcmc.plot_corner(uncorrelated_samples)
	fig.savefig('example_mcmc_corner.png')


	#### Save the analysis
	## Create a new group to hold the analysis in 'trajectory 0'
	dataset = b.smd.load(filename)
	trace = dataset['trajectory 0']
	mcmc_analysis = trace.create_group("MCMC analysis 20170106")

	## Add the priors
	b.smd.add.parameter_collection(mcmc_analysis,priors,label='priors')

	## Extract the relevant information from the sampler, and save this in the SMD file.
	result = b.mcmc.mcmc_result(sampler)
	b.smd.add.mcmc(mcmc_analysis,result,label='MCMC posterior samples')

	## Save and close the dataset
	b.smd.save(dataset)
	

Laplace approximation and computing the predictive posterior
---------------------------------------------------------------

This script loads the example data, sets some priors, and then finds the Laplace approximation to the posterior distribution. After this, it uses samples from this posterior to calculate the predictive posterior, which is the probability distribution for where you would expect to find new data.

.. code-block:: python
	
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
	
