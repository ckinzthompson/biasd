.. _examples:

BIASD Examples
==============

Here are some example Python scripts to perform BIASD. They can be found in `./example_data`, along with simulated data in a tab-delimited format (`./example_data/example_data.dat`), and an example HDF5 SMD dataset containing this data and some analysis results (`./example_data/example_dataset.hdf5`).


Sample the posterior with MCMC
------------------------------
This script loads the example data from above, sets some priors, and then uses the Markov chain Monte Carlo (MCMC) technique to sample the posterior.

.. code-block:: python
	
	## Imports
	import matplotlib.pyplot as plt
	import numpy as np
	import biasd as b
	b.likelihood.use_python_ll()

	#### Setup the analysis
	import simulate_singlemolecules as ssm
	data = ssm.testdata(nmol=1,nt=5000).flatten()
	tau = 0.1

	#### Perform a Calculation
	## Make the prior distribution
	e1 = b.distributions.normal(0., 0.2)
	e2 = b.distributions.normal(1.0, 0.2)
	sigma = b.distributions.loguniform(1e-3,1e0)
	k1 = b.distributions.loguniform(tau*.1,tau*100.)
	k2 = b.distributions.loguniform(tau*.1,tau*100.)
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

	## Setup the MCMC sampler to use 20 walkers 
	nwalkers = 20
	sampler, initial_positions = b.mcmc.setup(data, prior, tau, nwalkers)

	## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=500, progress=True)

	## Run 2000 fresh steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=2000,progress=True)
	samples = b.mcmc.get_samples(sampler)
	np.save('samples.npy',samples)

	## Find the best sample (MAP)
	lnp = sampler.get_log_prob()
	best = sampler.get_chain()[np.where(lnp==lnp.max())][0]
	np.save('MAP.npy',best)
	

Laplace approximation and computing the predictive posterior
---------------------------------------------------------------

This script loads the example data, sets some priors, and then finds the Laplace approximation to the posterior distribution. After this, it uses samples from this posterior to calculate the predictive posterior, which is the probability distribution for where you would expect to find new data.

.. code-block:: python
	
	## Imports
		b.likelihood.use_python_ll()

	import simulate_singlemolecules as ssm
	data = ssm.testdata(nmol=1,nt=5000).flatten()
	tau = 0.1

	e1 = b.distributions.normal(0., 0.2)
	e2 = b.distributions.normal(1.0, 0.2)
	sigma = b.distributions.loguniform(1e-3,1e0)
	k1 = b.distributions.loguniform(tau*.1,tau*100.)
	k2 = b.distributions.loguniform(tau*.1,tau*100.)
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

	guess = np.array((0.,1.,.08,1.,2.))
	posterior = b.laplace.laplace_approximation(data,prior,tau,guess=guess,verbose=True,ensure=True)
	fig,ax = b.plot.laplace_hist(data,tau,posterior)
	plt.show()	
