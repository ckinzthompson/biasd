import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_mcmc():
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

	## Setup the MCMC sampler to use 100 walkers and 2 CPUs
	nwalkers = 20
	sampler, initial_positions = b.mcmc.setup(data, prior, tau, nwalkers)

	## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=500, progress=True)

	## Run 2000 fresh steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=2000,progress=True)
	b.mcmc.get_samples(sampler)
	b.mcmc.chain_statistics(sampler)

	## Show Histogram + likelihood
	fig,ax = b.plot.mcmc_hist(data,tau,sampler)
	fig.savefig(os.path.join(fdir,'fig_mcmc.png'))
	plt.close()

	## Show Corner
	fig = b.plot.mcmc_corner(sampler)
	fig.savefig(os.path.join(fdir,'fig_corner.png'))
	plt.close()

	## Check stats
	mu,std = b.mcmc.get_stats(sampler)
	truth = np.array((0.,1.,.05,3.,8.))
	delta = np.abs(mu-truth)
	target = np.array((.1,.1,.02,2.,2.))
	assert np.all(delta < target)

if __name__ == '__main__':
	test_mcmc()