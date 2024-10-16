'''
.. module:: titrations
	:synopsis: Adapt BIASDs mcmc to do concentration-dependent datasets

'''


import numpy as np
import emcee
from . import likelihood,distributions

class collection_titration(distributions.collection):	
	def __init__(self,e1, e2, sigma1, sigma2, ka, kd):
		super().__init__(e1=e1, e2=e2, sigma1=sigma1, sigma2=sigma2, ka=ka, kd=kd)
		self.fancy_labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma_1$', r'$\sigma_2$', r'$\k_a^\prime$', r'$\k_d$']
		self.check_dists()

def log_titration_posterior(theta, data, concs, prior, tau, device=0):
	"""
	Calculate the global log-posterior probability distribution at :math:`\\Theta`
	If using CUDA, you should have pre-loaded the data onto the GPU using `load_cuda`

	Input:
		* `theta` is a vector of the parameters (i.e., :math:`\\theta`) where to evaluate the log-posterior
		  in the order: e1, e2, sigma1, sigma2, ka, kd. this assumes k12 from 1 to 2 is the concentration dependent rate constant
		* `data` is a list of N 1D `np.ndarray`s of the time series at N temperature points to analyze
		* `concs` is a `np.ndarray` of N concentration points that `data` corresponds to
		* `prior` is a `collect_titration`
		* `tau` is the measurement period of `data`

	Returns:
		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
	"""

	# ensures that e1 < e2
	if theta[0] > theta[1]:
		return -np.inf

	lnprior = prior.lnpdf(theta)
	if np.isnan(lnprior):
		return -np.inf
	elif not np.isfinite(lnprior):
		return -np.inf

	y = lnprior
	for i in range(len(concs)):
		params = theta.copy()
		params[4] = theta[4]*concs[i]
		y += likelihood.log_likelihood(params,data[i],tau,device=device)

	if np.isnan(y):
		return -np.inf
	else:
		return y


def setup(data, concs, prior, tau, nwalkers, initialize='rvs', device=0):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a list of 5 1D `np.ndarray`s of the time series at 5 temperature points to analyze
		* `concs` is a `np.ndarray` of N concentration points that `data` corresponds to
		* `prior` is a `collection_titration`
		* `nwalkers` is the number of walkers in the MCMC ensemble. The more the better
		* `initialze` =
			- 'rvs' will initialize the walkers at a random spot chosen from the priors
			- 'mean' will initialize the walkers tightly clustered around the mean of the priors.
			- an (`nwalkers`,5) `np.ndarray` of whatever spots you want to initialize the walkers at.
		* `threads` is the number of threads to use for evaluating the log-posterior of the walkers. Be careful when using the CUDA log-likelihood function, because you'll probably be bottle-necked there.

	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""

	ndim = 6		#corresponding to e1, e2, sigma1, sigma2, ka, kd

	if isinstance(initialize,np.ndarray) and initialize.shape == (nwalkers,ndim):
		initial_positions = initialize

	elif initialize == 'rvs':
		initial_positions = prior.rvs(nwalkers).T

	elif initialize == 'mean':
		initial_positions = prior.mean()[None,:] + np.random.normal(size=(nwalkers,ndim))*1e-8

	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	# Slap-dash hackery to make sure the first E_fret is the lower one
	for i in range(initial_positions.shape[0]):
		if initial_positions[i,0] > initial_positions[i,1]:
			temp = initial_positions[i,0]
			initial_positions[i,0] = initial_positions[i,1]
			initial_positions[i,1] = temp

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_titration_posterior, args=[data,concs,prior,tau,device])

	return sampler,initial_positions