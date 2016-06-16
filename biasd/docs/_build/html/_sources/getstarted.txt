.. _getstarted:

Getting Started
===============

Here're some quick examples to get you started using BIASD. In general, BIASD uses the SMD data format (DOI: 10.1186/s12859-014-0429-4) for data storage, though this is not required. It also uses `emcee` (arXiv:1202.3665) to perform the Markov chain Monte Carlo (MCMC), though the Laplace approximation is also provided, which does not use emcee.

You can install ``emcee`` with 

.. code-block:: bash
	
	pip install emcee

You might also want to get ``corner`` for plotting purposes. Use

.. code-block:: bash
	
	pip install corner


BIASD + MCMC
------------
BIASD uses `emcee`, which is a seriously awesome, affine invariant Markov chain Monte Carlo sample. Read about it `here <http://dan.iel.fm/emcee/current/>`_. 

.. code-block:: Python

	import numpy as np
	import biasd as b

	# Load some SMD format data
	data = b.smd.load('data.smd')

	# Setup prior distributions
	e1 = b.distributions.beta(1,9.)
	e2 = b.distributions.beta(9.2,.8)
	sigma = b.distributions.gamma(1.,1./4.8)
	k1 = b.distributions.gamma(1.,1./3)
	k2 = b.distributions.gamma(1.,1./8.)

	# Collect the distributions
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

	# Loop over all the traces
	for i in range(data.attr.n_traces):

		# Log the priors for this molecule in the SMD
		data = b.smd.add.priors(data,i,priors)

		# Setup the MCMC sampler with 50 walkers and 8 CPUs on the FRET data
		nwalkers = 50
		sampler, initial_positions = b.mcmc.setup(data.data[i].values.FRET,
			b.smd.read.priors(data,i), tau, 
			nwalkers, initialize='rvs', threads=8)

		# Run the MCMC: burn-in first, then production
		sampler, burned_positions = b.mcmc.burn_in(sampler, 
			initial_positions, nsteps=100)
		sampler = b.mcmc.run(sampler,burned_positions,nsteps=1000,timer=True)

		# Calculate acceptance ratio and autocorrelation times
		largest_autocorrelation_time = b.mcmc.chain_statistics(sampler)

		# Save this data
		data = b.smd.add.mcmc(data,i,sampler)
		b.smd.save('data.smd',data)


Plot BIASD-MCMC Results Using Corner
------------------------------------

Use corner to plot the 5-D space of the posterior sampled by MCMC. Read about corner `here <http://corner.readthedocs.io/en/latest/>`_.

.. code-block:: Python

	import numpy as np
	import biasd as b
	
	# Load some SMD format data
	data = b.smd.load('data.smd')
	
	# Get read the sampler results for the first trace (0)
	sampler_results = b.smd.read.mcmc(s,0)
	
	# Get the correlated samples
	samples_corr = sampler_results.chain
	
	# Remove some really bad samples
	cut = sampler_results.lnprobability < 0.
	samples_corr = samples_corr[~cut].reshape((-1,5))
	
	# Get the uncorrelated samples (previously calculated)
	samples_uncorr = sampler_results.samples

	# Plot corner plots
	f = b.mcmc.plot_corner(samples_corr)
	f = b.mcmc.plot_corner(samples_uncorr)
	


Plot BIASD-MCMC Results Using Viewer
------------------------------------

From the above example with corner, you can use the built in distribution viewer to explore the marginalized BIASD posterior distribution.

.. code-block:: Python

	# Create a collection of distributions from the marginalized samples
	posterior = b.mcmc.create_posterior_collection(samples_uncorr,priors)

	# View the marginalized posterior in a biasd.distribution.viewer
	b.distributions.viewer(posterior)


BIASD + Laplace Approximation
-----------------------------

You can also use the Laplace approximation to approximate the posterior distribution as a multidimensional gaussian centered the the maximum a postiori (MAP) value of the distribution. The advantage is that it is probably faster than MCMC, however, it is definitely an approximation.

.. code-block:: Python

	import numpy as np
	import biasd as b
	
	# Load some SMD format data
	data = b.smd.load('data.smd')
	tau = 0.1
	
	# Setup the prior distributions 
	e1 = b.distributions.beta(1.,9.)
	e2 = b.distributions.beta(9.,1.)
	sigma = b.distributions.gamma(2.,2./.05)
	k1 = b.distributions.gamma(20.,20./3.)
	k2 = b.distributions.gamma(20.,20./8.)
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

	# Loop over the traces
	for i in range(data.attr.n_traces):
		
		# Perform Laplace approximation
		d = data.data[i].values.FRET
		lp = b.laplace.laplace_approximation(d,priors,tau)

		# Add the priors and results to the SMD
		data = b.smd.add.priors(data,i,priors)
		data = b.smd.add.laplace_posterior(data,i,lp)

		# Calculate the moment-matched, marginalized posterior
		# of the same form as the prior distributions
		lp.transform(priors)
		data = b.smd.add.posterior(data,i,lp.posterior)
	
	# Save the results
	b.smd.save('data.smd',data)


My Baseline and BIASD...
------------------------
If your baseline is crazy, BIASD will not work very well. In the ``./utils`` directory there a method to try to integrate out a baseline that follows a Brownian diffusion process. Since this implementation is built upon a Gaussian mixture model, it's probably inappropriate to use this when there is a lot of blurring.

.. code-block:: Python

	# Load data
	data = b.smd.load('data.smd')
	
	# Let's remove the baseline of the first trace
	d = data.data[0].values.FRET
	baseline_result = b.baseline.remove_baseline(d)
	data = b.smd.add.baseline(data,0,baseline_result)

	# Save the results
	b.smd.save('data.smd',data)

	# Subtract off the baseline for use in some other calculation
	d -= baseline_result.baseline 
	
