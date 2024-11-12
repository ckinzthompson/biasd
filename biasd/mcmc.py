"""
.. module:: mcmc
	:synopsis: Integrates emcee's MCMC into BIASD

"""


import numpy as np
import emcee
import time

from . import likelihood
from . import distributions

def setup(data, priors, tau, nwalkers, initialize='rvs', device=0, backend=None):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a `np.ndarray` of the time series
		* `priors` is a `biasd.distributions.parameter_collection` of the priors
		* `tau` is the measurement period each data point
		* `nwalkers` is the number of walkers in the MCMC ensemble. The more the better
		* `initialze` =
			- 'rvs' will initialize the walkers at a random spot chosen from the priors
			- 'mean' will initialize the walkers tightly clustered around the mean of the priors.
			- an (`nwalkers`,5) `np.ndarray` of whatever spots you want to initialize the walkers at.

	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""


	ndim = priors.num

	if isinstance(initialize,np.ndarray) and initialize.shape == (nwalkers,ndim):
		initial_positions = initialize
	elif initialize == 'rvs':
		initial_positions = priors.rvs(nwalkers).T
	elif initialize == 'mean':
		initial_positions = priors.mean()[None,:] + np.random.normal(size=(nwalkers,ndim))*1e-8
	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	## Make sure the first state epsilon is the smaller
	for i in range(initial_positions.shape[0]):
		if initial_positions[i,0] > initial_positions[i,1]:
			temp = initial_positions[i,0]
			initial_positions[i,0] = initial_positions[i,1]
			initial_positions[i,1] = temp

	sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood.log_posterior, args=[data,priors,tau,device],backend=backend)

	return sampler,initial_positions

def burn_in(sampler,positions,nsteps,progress=True):
	"""
	Burn-in will run some MCMC, getting new positions, and then reset the sampler so that nothing has been sampled.

	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `progress` is a boolean for displaying the timing statistics

	Results:
		* `sampler` is now a cleared `emcee` sampler where no steps have been made
		* `positions` is an array of the final walkers positions for use when starting a more randomized sampling

	"""
	sampler = run(sampler,positions,nsteps,progress)
	positions = np.copy(sampler.get_last_sample().coords)
	sampler.reset()
	return sampler,positions

def run(sampler,positions,nsteps,progress=True):
	"""
	Acquire some MCMC samples, and keep them in the sampler

	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `progress` is a boolean for displaying the timing statistics

	Results:
		* `sampler` is the updated `emcee` sampler

	"""

	t0 = time.time()
	sampler.run_mcmc(positions,nsteps,progress=progress)
	t1 = time.time()
	if progress:
		print("Steps: ", sampler.get_chain().shape[0])
		print("Total Time:",(t1-t0))
		print("Time/Sample:",(t1-t0)/sampler.nwalkers/nsteps)
	return sampler

def continue_run(sampler,nsteps,progress=True):
	"""
	Similar to `biasd.mcmc.run`, but you do not need to specify the initial positions, because they will be the last sampled positions in `sampler`
	"""
	positions = np.copy(sampler.get_last_sample().coords)
	sampler = run(sampler,positions,nsteps,progress=progress)
	return sampler

def chain_statistics(sampler,verbose=False):
	"""
	Calculate the acceptance fraction and autocorrelation times of the samples in `sampler`
	"""
	# Chain statistics
	act = sampler.get_autocorr_time(quiet=True)
	if verbose:
		print("Mean Acceptance fraction:", np.mean(sampler.acceptance_fraction))
		print("Autocorrelation time:", act)
	maxauto = act.astype('int').max()+1
	return maxauto

def get_samples(sampler,uncorrelated=True,culled=False,verbose=False):
	"""
	Get the samples from `sampler`

	Input:
		* `sampler` is an `emcee` sampler with samples in it
		* `uncorrelated` is a boolean for whether to provide all the samples, or every n'th sample, where n is the larges autocorrelation time of the dimensions.
		* `culled` is a boolean, where any sample with a log-probability less than 0 is removed. This is necessary because sometimes a few chains get very stuck, and their samples (not being representative of the posterior) mess up subsequent plots.

	Returns:
		An (N,5) `np.ndarray` of samples from the sampler
	"""

	if uncorrelated:
		maxauto = chain_statistics(sampler,verbose=verbose)
	else:
		maxauto = 1
	if culled:
		keep = sampler.get_log_prob().mean(0) > 0.
	else:
		keep = np.isfinite(sampler.get_log_prob().mean(0))
	samples = sampler.get_chain()[::maxauto,keep,:]
	samples = samples.reshape((samples.size//sampler.ndim,sampler.ndim))
	return samples

def get_stats(sampler):
	ss = get_samples(sampler,uncorrelated=True)
	mu = ss.mean(0)
	std = ss.std(0)

	print(f'No. Samples: {ss.shape[0]}')
	# labels = ['e1 ','e2 ','sig','k1 ','k2 ']
	for i in range(mu.size):
		# print(f"{labels[i]}: {mu[i]:.4f} +/- {std[i]:.4f}")
		print(f"Parameter {i}: {mu[i]:.4f} +/- {std[i]:.4f}")
	return mu,std

def create_posterior_collection(samples,priors):
	"""
	Take the MCMC samples, marginalize them, and then calculate the first and second moments. Use these to moment-match to the types of distributions specified for each dimension in the priors. For instance, if the prior for :math:`\\epsilon_1` was beta distributed, this will moment-match the posterior to as a beta distribution.

	Input:
		* `samples` is a (N,5) `np.ndarray`
		* `priors` is a `biasd.distributions.parameter_collection` that provides the distribution-forms to moment-match to
	Returns:
		* A `biasd.distributions.parameter_collection` containing the marginalized, moment-matched posteriors
	"""

	#Moment-match, marginalized posteriors
	first = samples.mean(0)
	second = np.var(samples,axis=0)+first**2

	e1 = priors.e1.new(np.around(priors.e1._moment2param_fxn(first[0], second[0]),4))
	e2 = priors.e2.new(np.around(priors.e2._moment2param_fxn(first[1], second[1]),4))
	if first.size == 5:
		sigma = priors.sigma.new(np.around(priors.sigma._moment2param_fxn(first[2], second[2]),4))
		k1 = priors.k1.new(np.around(priors.k1._moment2param_fxn(first[3], second[3]),4))
		k2 = priors.k2.new(np.around(priors.k2._moment2param_fxn(first[4], second[4]),4))
		return distributions.collection_standard_1sigma(e1,e2,sigma,k1,k2)
	elif first.size==6:
		sigma1 = priors.sigma.new(np.around(priors.sigma._moment2param_fxn(first[2], second[2]),4))
		sigma2 = priors.sigma.new(np.around(priors.sigma._moment2param_fxn(first[3], second[3]),4))
		k1 = priors.k1.new(np.around(priors.k1._moment2param_fxn(first[4], second[4]),4))
		k2 = priors.k2.new(np.around(priors.k2._moment2param_fxn(first[5], second[5]),4))
		return distributions.collection_standard_2sigma(e1,e2,sigma1,sigma2,k1,k2)
	else:
		raise Exception("Not Implemented")

class mcmc_result(object):
	"""
	Holds the results of a MCMC sampler of the posterior probability distribution from BIASD
	Input:
		* `mcmc_input` is either an `emcee.sampler.Sampler` or child, or a list of `[acor, chain, lnprobability, iterations, naccepted, nwalkers, dim]`
	"""
	def __init__(self, mcmc_input):
		try:
			if 'lnprobfn' in mcmc_input.__dict__:
				if 'acor' not in mcmc_input.__dict__:
					mcmc_input.get_autocorr_time()
				self.acor = mcmc_input.get_autocorr_time()
				self.chain = mcmc_input.get_chain()
				self.lnprobability = mcmc_input.get_log_prob()
				self.iterations = mcmc_input.iterations
				self.naccepted = mcmc_input.naccepted
				self.nwalkers = mcmc_input.k
				self.dim = mcmc_input.dim
				return
		except:
			pass
		try:
			self.acor, self.chain, self.lnprobability, self.iterations, self.naccepted,self.nwalkers,self.dim = mcmc_input
			return
		except:
			pass
		raise Exception("Couldn't initialize mcmc_result")
