'''
.. module:: constant epsilon
	:synopsis: Adapt BIASDs mcmc to do concentration-dependent datasets

'''


import numpy as np
import emcee
from . import likelihood,distributions

class collection_constantepsilonsigma(distributions.collection):	
	'''
	kk_list is [(k1_1,k2_1),(k1_2,k2_2),...]
	'''
	def __init__(self, e1, e2, sigma, kk_list):
		super().__init__(e1=e1, e2=e2, sigma=sigma)

		nkk = len(kk_list)
		for i in range(nkk):
			labeli = [f'k1_{i}',f'k2_{i}']
			for label,param in zip(labeli,kk_list[i]):
				setattr(self,label,param)
				self.parameters[label] = param
			self.labels += labeli
			self.num = len(self.labels)
		self.ndim = nkk
		self.check_dists()

def log_constantepsilonsigma_posterior(theta, data, prior, tau, device=0):
	"""
	Calculate the global log-posterior probability distribution at :math:`\\Theta`
	If using CUDA, you should have pre-loaded the data onto the GPU using `load_cuda`

	Input:
		* `theta` is a vector of the parameters (i.e., :math:`\\theta`) where to evaluate the log-posterior
		* `data` is a list of N 1D `np.ndarray`s of the time series at N temperature points to analyze
		* `prior` is a `collect_constantepsilon`
		* `tau` is the measurement period of `data`

	Returns:
		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
	"""

	# ensures that e1 < e2
	if theta[0] > theta[1]:
		return -np.inf

	if ((theta.size-3)//2) != prior.ndim:
		raise Exception('Malformed prior or theta')
	
	if np.any(theta[3:]*tau) < 1e-16: ## it's numerically zero (or negative!)
		return -np.inf 


	lnprior = prior.lnpdf(theta)
	if np.isnan(lnprior):
		return -np.inf
	elif not np.isfinite(lnprior):
		return -np.inf

	y = lnprior
	for i in range(prior.ndim):
		params = np.concatenate((theta[:3],theta[3+2*i:3+2*(i+1)]))
		y += likelihood.log_likelihood(params,data[i],tau,device=device)
	
	if np.isnan(y):
		return -np.inf
	else:
		return y


def setup(data, prior, tau, nwalkers, initialize='rvs', device=0, backend=None):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a list of 5 1D `np.ndarray`s of the time series at 5 temperature points to analyze
		* `prior` is a `collection_constantepsilon`
		* `nwalkers` is the number of walkers in the MCMC ensemble. The more the better
		* `initialze` =
			- 'rvs' will initialize the walkers at a random spot chosen from the priors
			- 'mean' will initialize the walkers tightly clustered around the mean of the priors.
			- an (`nwalkers`,5) `np.ndarray` of whatever spots you want to initialize the walkers at.
		* `threads` is the number of threads to use for evaluating the log-posterior of the walkers. Be careful when using the CUDA log-likelihood function, because you'll probably be bottle-necked there.

	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""

	ndim = prior.ndim*2+3		#corresponding to e1, e2, sigma, {k1,k2}_i,...

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

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_constantepsilonsigma_posterior, args=[data,prior,tau,device],backend=backend)

	return sampler,initial_positions