.. _code_mcmc:

MCMC
====

This page gives the details about the code in biasd.mcmc.

Markov chain Monte Carlo
++++++++++++++++++++++++

To sample the posterior probability distribution in BIASD, we'll use an affine invariant Markov chain Monte Carlo (MCMC) sampler. The implementation here uses `emcee <http://dan.iel.fm/emcee/current/>`_, which allow very efficient MCMC sampling. It is described in

:Title:
	emcee: The MCMC Hammer
:Authors:
	Daniel Foreman-Mackey,
	David W. Hogg,
	Dustin Lang,
	and Jonathan Goodman
:arXiv:
	http://arxiv.org/abs/1202.3665
:DOI:
	10.1086/670067

Which extends upon the paper

:Title:
	Ensemble samplers with affine invariance
:Authors:
	Jonathan Goodman,
	and Jonathan Weare
:Citation:
	*Comm. Appl. Math. Comp. Sci.* **2010**, *5(1)*, 65-80.
:DOI:
	10.2140/camcos.2010.5.65>


Setup and Run MCMC
++++++++++++++++++

.. automodule:: mcmc
	:members: setup, burn_in, run, continue_run

Example use:

.. code-block:: python

	import biasd as b
	
	# Load data
	data = b.smd.load('data.smd')
	tau = 0.1
	
	# Get a molecule and priors
	d = data.data[0].values.FRET
	priors = b.distributions.guess_priors(d,tau)
	
	# Setup the sampler for this molecule
	# Use 100 walkers, and 4 CPUs
	sampler, initial_positions = b.mcmc.setup(dy, priors, tau, 100, initialize='rvs', threads=4)

	# Burn-in 100 steps and then remove them, but keep the final positions
	sampler,burned_positions = b.mcmc.burn_in(sampler,initial_positions,nsteps=100)
	
	# Run 100 steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=100)
	# Continue on from step 100 for another 900 steps. Don't display timing
	sampler = b.mcmc.continue_run(sampler,900,timer=False)
	
	# Save the sampler data
	data = b.smd.add.mcmc(sampler)
	b.smd.save('data.smd',data)

Analyze MCMC samples
++++++++++++++++++++

.. automodule:: mcmc
	:members: chain_statistics, get_samples, plot_corner, create_posterior_collection

Note, for the corner plot, you must have corner. Anyway, continuing on from the previous example...

Example:

.. code-block:: python

	# ...
	
	# Calculate auto-correlation times for each variable
	largest_autocorrelation_time = b.mcmc.chain_statistics(sampler)

	# Collect uncorrelated samples from the sampler
	samples = b.mcmc.get_samples(sampler)
	
	# Plot the joint and marginalized distributions from the samples using corner, and then save the figure
	f = b.mcmc.plot_corner(samples)
	plt.savefig('mcmc_test.pdf')

	# Create a collection of the marginalized posterior distributions
	posterior = b.mcmc.create_posterior_collection(samples,priors)

	# View that collection
	b.distributions.viewer(posterior)