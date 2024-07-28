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
	import matplotlib.pyplot as plt
	import numpy as np

	b.likelihood.use_python_ll()

	## Simulate data
	import simulate_singlemolecules as ssm
	data = ssm.testdata(nmol=5,nt=100).flatten()
	tau = .1

	## Setup prior
	e1 = b.distributions.normal(0.,.01)
	e2 = b.distributions.normal(1.,.01)
	sigma = b.distributions.loguniform(.01,.1)
	k12 = b.distributions.loguniform(1.,30.)
	k21 = b.distributions.loguniform(1.,30.)
	prior = b.distributions.parameter_collection(e1,e2,sigma,k12,k21)

	## Setup the MCMC sampler to use 20 walkers 
	nwalkers = 20
	sampler, initial_positions = b.mcmc.setup(data, prior, tau, nwalkers)

	## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=500, progress=True)

	## Run 2000 fresh steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=2000,progress=True)
	gamples = b.mcmc.get_samples(sampler)
	b.mcmc.chain_statistics(sampler)

Analyze MCMC samples
++++++++++++++++++++

.. automodule:: mcmc
	:members: chain_statistics, get_samples, plot_corner, create_posterior_collection

Note, for the corner plot, you must have corner. Anyway, continuing on from the previous example...

Example:

.. code-block:: python

	# ...
	
	## Show Histogram + likelihood
	fig,ax = b.plot.mcmc_hist(data,tau,sampler)
	fig.savefig('./fig_mcmc.png')
	plt.show()

	## Show Corner
	fig = b.plot.mcmc_corner(sampler)
	plt.savefig('fig_corner.png')
	plt.show()