'''
.. module:: temperature
	:synopsis: Adapt BIASDs mcmc to do temperature-dependent datasets

'''


import numpy as np
import emcee
from . import likelihood,distributions

class collection_temperature(distributions.collection):	
	def __init__(self,e1, e2, sigma, H1, S1, H2, S2):
		super().__init__(e1=e1, e2=e2, sigma=sigma, H1=H1, S1=S1, H2=H2, S2=S2)
		self.fancy_labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma$', r'$\Delta H_1$',r'$\Delta S_1$',r'$\Delta H_2$',r'$\Delta S_2$']
		self.check_dists()

def TST(dH,dS,T):
	# the following parameters for the TST equation are in SI units
	## dH (joules/mol) and dS (joules/mol/K)
	kappa = 1
	kB = 1.38064852e-23
	h = 6.62607004e-34
	R = 8.314

	lnk = np.log(kappa*kB*T/h) + dS/R - dH/(R*T)
	k = np.exp(lnk)
	return k

def log_temperature_posterior(theta, data, T, prior, tau, device=0):
	"""
	Calculate the global log-posterior probability distribution at :math:`\\Theta`
	If using CUDA, you should have pre-loaded the data onto the GPU using `load_cuda`

	Input:
		* `theta` is a vector of the parameters (i.e., :math:`\\theta`) where to evaluate the log-posterior
		  in the order: e1, e2, sigma, H1, S1, H2, S2
		* `data` is a list of N 1D `np.ndarray`s of the time series at N temperature points to analyze
		* `T` is a `np.ndarray` of N temperature points that `data` corresponds to
		* prior is a `collection_tempeterature` which is a subclass of `distributions.collection`
		* `tau` is the measurement period of `data`

	Returns:
		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
	"""

	# ensures that e1 < e2
	if theta[0] > theta[1]:
		return -np.inf

	H1, S1, H2, S2 = theta[3:]
	k1 = TST(H1,S1,T)
	k2 = TST(H2,S2,T)

	lnprior = prior.lnpdf(theta)
	if np.isnan(lnprior):
		return -np.inf
	elif not np.isfinite(lnprior):
		return -np.inf

	y = lnprior
	for i in range(len(T)):
		params = np.concatenate((theta[:3], np.array([k1[i], k2[i]])))
		if likelihood.ll_version == "CUDA":
			y += likelihood.log_likelihood(params,i,tau,device=device)
		else:
			y += likelihood.log_likelihood(params,data[i],tau,device=device)

	if np.isnan(y):
		return -np.inf
	else:
		return y


def setup(data, T, prior, tau, nwalkers, initialize='rvs', device=0):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a list of 5 1D `np.ndarray`s of the time series at 5 temperature points to analyze
		* `T` is a `np.ndarray` of 5 temperature points that `data` corresponds to
		* prior is a `collection_tempeterature` which is a subclass of `distributions.collection`
		* `tau` is the measurement period of `data`
		* `nwalkers` is the number of walkers in the MCMC ensemble. The more the better
		* `initialze` =
			- 'rvs' will initialize the walkers at a random spot chosen from the priors
			- 'mean' will initialize the walkers tightly clustered around the mean of the priors.
			- an (`nwalkers`,5) `np.ndarray` of whatever spots you want to initialize the walkers at.
		* `threads` is the number of threads to use for evaluating the log-posterior of the walkers. Be careful when using the CUDA log-likelihood function, because you'll probably be bottle-necked there.

	Results:
		* An `emcee` sampler object. Please see the `emcee` documentation for more information.
	"""

	ndim = 7		#corresponding to e1, e2, sigma, H1, S1, H2, S2

	if isinstance(initialize,np.ndarray) and initialize.shape == (nwalkers,7):
		initial_positions = initialize

	elif initialize == 'rvs':
		initial_positions = prior.rvs(nwalkers).T
		H1 = initial_positions[:,3]
		S1 = (H1 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S1 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		initial_positions[:,4] = S1
		H2 = initial_positions[:,5]
		S2 = (H2 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S2 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		initial_positions[:,6] = S2

	elif initialize == 'mean':
		H1 = prior.parameters['H1'].mean()
		S1 = (H1 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S1 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		H2 = prior.parameters['H2'].mean()
		S2 = (H2 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S2 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		initial_positions = np.array([prior.mean(),]*nwalkers)
		initial_positions[:,4] = S1
		initial_positions[:,6] = S2

	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	# Slap-dash hackery to make sure the first E_fret is the lower one
	for i in range(initial_positions.shape[0]):
		if initial_positions[i,0] > initial_positions[i,1]:
			temp = initial_positions[i,0]
			initial_positions[i,0] = initial_positions[i,1]
			initial_positions[i,1] = temp

	# Loading the data on the GPU to make it persistent throughout the calculation
	if likelihood.ll_version == "CUDA":
		likelihood.free_cuda(device)
		likelihood.load_cuda(data,device)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_temperature_posterior, args=[data,T,prior,tau,device])

	return sampler,initial_positions