'''
.. module:: mcmc
	:synopsis: Integrates emcee's MCMC into BIASD

'''

import numpy as _np
import emcee
from time import time as _time

def setup(data, priors, tau, nwalkers, initialize='rvs', threads=1):
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
		* `threads` is the number of threads to use for evaluating the log-posterior of the walkers. Be careful when using the CUDA log-likelihood function, because you'll probably be bottle-necked there.
	
	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""
	
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
	"""
	Burn-in will run some MCMC, getting new positions, and then reset the sampler so that nothing has been sampled.
	
	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `timer` is a boolean for displaying the timing statistics
	
	Results:
		* `sampler` is now a cleared `emcee` sampler where no steps have been made
		* `positions` is an array of the final walkers positions for use when starting a more randomized sampling
		
	"""
	sampler = run(sampler,positions,nsteps,timer)
	positions = _np.copy(sampler.chain[:,-1,:])
	sampler.reset()
	return sampler,positions

def run(sampler,positions,nsteps,timer=True):
	"""
	Acquire some MCMC samples, and keep them in the sampler
	
	Input:
		* `sampler` is an `emcee` sampler
		* `positions` is the starting walker positions (maybe provided by `biasd.mcmc.setup`?)
		* `nsteps` is the integer number of MCMC steps to take
		* `timer` is a boolean for displaying the timing statistics
	
	Results:
		* `sampler` is the updated `emcee` sampler
	
	"""
	
	t0 = _time()
	sampler.run_mcmc(positions,nsteps)
	t1 = _time()
	if timer:		
		print "Steps: ", sampler.chain.shape[1]
		print "Total Time:",(t1-t0)
		print "Time/Sample:",(t1-t0)/sampler.flatchain.shape[0]/sampler.args[0].size
	return sampler

def continue_run(sampler,nsteps,timer=True):
	"""
	Similar to `biasd.mcmc.run`, but you do not need to specify the initial positions, because they will be the last sampled positions in `sampler`
	"""
	positions = sampler.chain[:,-1,:]
	sampler = run(sampler,positions,nsteps,timer=timer)
	return sampler
	

def chain_statistics(sampler,verbose=True):
	"""
	Calculate the acceptance fraction and autocorrelation times of the samples in `sampler`
	"""
	# Chain statistics
	if verbose:
		print "Mean acceptance fraction:", _np.mean(sampler.acceptance_fraction)
		print "Autocorrelation time:", sampler.get_autocorr_time()
	maxauto = _np.int(sampler.get_autocorr_time().max())+1
	return maxauto
	
def get_samples(sampler,uncorrelated=True,culled=False):
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
	"""
	Use the python package called corner <https://github.com/dfm/corner.py> to make some very nice corner plots (joints and marginalized) of posterior in the 5-dimensions used by the two-state BIASD posterior.
	
	Input:
		* `samples` is a (N,5) `np.ndarray`
	Returns:
		* `fig` which is the handle to the figure containing the corner plot
	"""
	
	import corner
	labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma$', r'$k_1$', r'$k_2$']
	fig = corner.corner(samples, labels=labels, quantiles=[.025,.50,.975],levels=(1-_np.exp(-0.5),))
	return fig

def create_posterior_collection(samples,priors):
	"""
	Take the MCMC samples, marginalize them, and then calculate the first and second moments. Use these to moment-match to the types of distributions specified for each dimension in the priors. For instance, if the prior for :math:`\\epsilon_1` was beta distributed, this will moment-match the posterior to as a beta distribution.
	
	Input:
		* `samples` is a (N,5) `np.ndarray`
		* `priors` is a `biasd.distributions.parameter_collection` that provides the distribution-forms to moment-match to
	Returns:
		* A `biasd.distributions.parameter_collection` containing the marginalized, moment-matched posteriors
	"""
	
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

class _mcmc_result(object):
	"""
	Holds the results of a MCMC sampler of the posterior probability distribution from BIASD
	Input:
		* `mcmc_input` is either an `emcee.sampler.Sampler` or child, or a list of `[acor, chain, lnprobability, iterations, naccepted, nwalkers, dim]`
	"""
	def __init__(self, mcmc_input):
		try:
			if mcmc_input.__dict__.has_key('lnprobfn'):
				if not mcmc_input.__dict__.has_key('acor'):
					mcmc_input.get_autocorr_time()
				self.acor = mcmc_input.acor
				self.chain = mcmc_input.chain
				self.lnprobability = mcmc_input.lnprobability
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
		raise Exception("Couldn't initialize _mcmc_result")
		