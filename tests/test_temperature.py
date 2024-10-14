import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import corner
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_temperature():
	######## Simulate Temperature-dependent datasets
	#### RC^{fMet} (L1) - see Ray et al, Entropic...
	## https://www.pnas.org/doi/10.1073/pnas.2220591120
	## SI Table S2 -- convert kcal (or cal) into J
	dHGS1 = 21.5 ## kcal/mol
	dSGS1 = 0.015 ## kcal/mol/K
	dHGS2 = 10.2
	dSGS2 = -0.0194

	## SI Table S3
	## NOTE: ep1 < ep2 in BIASD, so GS1/GS2 is sort of flipped here
	ep1 = 0.35
	ep2 = 0.55
	emissions = np.array((ep1,ep2))

	#### Other info
	temperatures = np.array((295.,300.,305.,310.,))
	k2 = b.temperature.TST(dHGS1*4184.,dSGS1*4184.,temperatures) ## fxn takes joules
	k1 = b.temperature.TST(dHGS2*4184.,dSGS2*4184.,temperatures)

	noise = .02
	nframes = 600
	tau = .1
	nmol = 2

	#### Simulation
	datasets = []
	for i in range(temperatures.size):
		temp = temperatures[i]
		rates = np.array(((0.,k1[i]),(k2[i],0.)))
		data = ssm.simulate_ensemble(rates,emissions,noise,nframes,tau,nmol)
		datasets.append(data.flatten())
	# 	plt.hist(data.flatten(),bins=201,range=(0,1),density=True,alpha=.3)
	# plt.show()

	######## Run Calculation
	b.likelihood.use_python_ll()

	#### Setup priors
	p_ep1 = b.distributions.normal(ep1,.01)
	p_ep2 = b.distributions.normal(ep2,.01)
	p_sigma = b.distributions.loguniform(noise/2.,2.*noise)

	## Units are joules/mol, joules/mol/K
	p_H1 = b.distributions.uniform(-500000., 500000.)
	p_S1 = b.distributions.uniform(-1000., 1000.)
	p_H2 = b.distributions.uniform(-500000., 500000.)
	p_S2 = b.distributions.uniform(-1000., 1000.)

	priors = b.temperature.collection_temperature(p_ep1,p_ep2,p_sigma,p_H1, p_S1, p_H2, p_S2)

	######## Temperature MCMC
	#### Note: The samples are in SI units.
	#### Setup with twice as many
	nwalkers = 100
	nsteps = 100
	sampler, initial_positions = b.temperature.setup(datasets, temperatures, priors, tau, nwalkers)
	sampler = b.mcmc.run(sampler, initial_positions, nsteps, progress=True)

	## Selecting the best half of the walkers to remove issues with initialisations
	last = sampler.get_last_sample()
	lx = last.log_prob.argsort()
	better_positions = last.coords[lx][-int(nwalkers//2):]

	#### Setting up the MCMC run with the better initialisations
	nwalkers = nwalkers // 2 
	sampler, _ = b.temperature.setup(datasets, temperatures, priors, tau, nwalkers,)

	## Burn in for the production run
	nburn = 500
	sampler, burned_positions = b.mcmc.burn_in(sampler, better_positions, nburn, progress=True)

	## Production run
	nprod = 2000
	sampler = b.mcmc.run(sampler, burned_positions, nprod, progress=True)

	## Make Corner plot
	b.mcmc.chain_statistics(sampler)
	samples = b.mcmc.get_samples(sampler, uncorrelated=True, verbose=False)
	samples[:,3:] /= 4184. ## convert from joules to kcal
	fig = corner.corner(samples)
	fig.savefig(os.path.join(fdir,'fig_temperature.png'))
	plt.close()

	## Get the MAP solution
	lnp = sampler.get_log_prob()
	best = sampler.get_chain()[np.where(lnp==lnp.max())]
	if best.ndim > 1:
		best = best[0]
	best[3:]/= 4184. ## convert from joules to kcal

	## Check results
	truth = np.array((ep1,ep2,noise,dHGS2,dSGS2,dHGS1,dSGS1))
	delta = np.abs(best-truth)
	target = np.array((.01,.01,.01,5.,.01,5.,.01))
	print(best)
	print(truth)
	print(delta)
	print(delta<target)
	assert np.all(delta < target)

if __name__ == '__main__':
	test_temperature()