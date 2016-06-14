'''
Helper functions for running affine invariant Markov chain Monte Carlo on BIASD posteriors

--------------------

Uses the emcee package: http://dan.iel.fm/emcee/current/

As described in:
emcee: The MCMC Hammer
Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, and Jonathan Goodman
http://arxiv.org/abs/1202.3665
DOI:10.1086/670067

Based upon:
Ensemble samplers with affine invariance
Jonathan Goodman and Jonathan Weare
Comm. Appl. Math. Comp. Sci. 2010, 5(1), 65-80.
DOI: 10.2140/camcos.2010.5.65

---------------

Example use:

sampler,initial_positions = b.mcmc.setup(dy, priors, tau, 16, initialize='rvs', threads=20)

sampler,burned_positions = b.mcmc.burn_in(sampler,initial_positions,nsteps=50)
sampler = b.mcmc.run(sampler,burned_positions,nsteps=100,timer=False)
sampler = b.mcmc.continue_run(sampler,900)

largest_autocorrelation_time = b.mcmc.chain_statistics(sampler)

samples = b.mcmc.get_samples(sampler)
f = b.mcmc.plot_corner(samples)
plt.savefig('mcmc_test.png')

posterior = b.mcmc.create_posterior_collection(samples,priors)
b.distributions.viewer(posterior)
'''

import numpy as _np
import emcee
from time import time as _time

def setup(data, priors, tau, nwalkers, initialize='rvs', threads=1):
	from biasd.likelihood import log_posterior
	ndim = 5
	
	if isinstance(initialize,_np.ndarray) and initialize.shape == (nwalkers,5):
		initial_positions = initialize
	elif initialize == 'rvs':
		initial_positions = priors.rvs(nwalkers).T
	elif initialize == 'mean':
		initial_positions = _np.array([priors.mean()+1e-6*_np.random.rand(5) for _ in range(nwalkers)])
	
	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[data,priors,tau],threads=threads)
	
	return sampler,initial_positions

def burn_in(sampler,positions,nsteps=100,timer = True):
	sampler = run(sampler,positions,nsteps,timer)
	positions = _np.copy(sampler.chain[:,-1,:])
	sampler.reset()
	return sampler,positions

def run(sampler,positions,nsteps,timer=True):
	t0 = _time()
	sampler.run_mcmc(positions,nsteps)
	t1 = _time()
	if timer:		
		print "Steps: ", sampler.chain.shape[1]
		print "Total Time:",(t1-t0)
		print "Time/Sample:",(t1-t0)/sampler.flatchain.shape[0]/sampler.args[0].size
	return sampler

def continue_run(sampler,nsteps,timer=True):
	positions = sampler.chain[:,-1,:]
	sampler = run(sampler,positions,nsteps,timer=timer)
	return sampler
	

def chain_statistics(sampler,verbose=True):
	# Chain statistics
	if verbose:
		print "Mean acceptance fraction:", _np.mean(sampler.acceptance_fraction)
		print "Autocorrelation time:", sampler.get_autocorr_time()
	maxauto = _np.int(sampler.get_autocorr_time().max())+1
	return maxauto
	
def get_samples(sampler,uncorrelated=True,culled=True):
	if uncorrelated:
		maxauto = chain_statistics(sampler,verbose=False)
	else:
		maxauto = 1
	if culled:
		cut = sampler.lnprobability.mean(1) < 0.
	else:
		cut = sampler.lnprobability.mean(1) < -_np.inf
	samples = sampler.chain[~cut,::maxauto,:].reshape((-1,5))
	return samples

def plot_corner(samples):
	import corner
	labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma$', r'$k_1$', r'$k_2$']
	fig = corner.corner(samples, labels=labels, quantiles=[.025,.50,.975],levels=(1-_np.exp(-0.5),))
	return fig

def create_posterior_collection(samples,priors):
	from biasd.distributions import parameter_collection
	#Moment-match, marginalized posteriors
	first = samples.mean(0)
	second = _np.var(samples,axis=0)+first**2
	
	e1 = priors.e1.new(_np.around(priors.e1._moment2param_fxn(first[0], second[0]),4))
	e2 = priors.e2.new(_np.around(priors.e2._moment2param_fxn(first[1], second[1]),4))
	sigma = priors.sigma.new(_np.around(priors.sigma._moment2param_fxn(first[2], second[2]),4))
	k1 = priors.k1.new(_np.around(priors.k1._moment2param_fxn(first[3], second[3]),4))
	k2 = priors.k2.new(_np.around(priors.k2._moment2param_fxn(first[4], second[4]),4))
	
	return parameter_collection(e1,e2,sigma,k1,k2)
	