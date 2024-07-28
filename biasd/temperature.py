'''
.. module:: mcmc
	:synopsis: Adapt BIASDs mcmc to do temperature-dependent datasets

'''


import numpy as np
import emcee
from .likelihood import log_likelihood, ll_version
from .mcmc import burn_in, run, continue_run, chain_statistics, get_samples

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

def log_temperature_posterior(theta, data, T, theta_prior, E_priors, tau, device=0):
	"""
	Calculate the global log-posterior probability distribution at :math:`\\Theta`
	If using CUDA, you should have pre-loaded the data onto the GPU using `load_cuda`

	Input:
		* `theta` is a vector of the parameters (i.e., :math:`\\theta`) where to evaluate the log-posterior
		  in the order: e1, e2, sigma, H1, S1, H2, S2
		* `data` is a list of N 1D `np.ndarray`s of the time series at N temperature points to analyze
		* `T` is a `np.ndarray` of N temperature points that `data` corresponds to
		* `theta_prior` is a `biasd.distributions.parameter_collection` containing the prior probability
		  distributions for e1, e2, sigma (along with fake k1 and k2 priors) for the BIASD calculation
		* `E_priors` is a list of probability distributions drawn from `biasd.distributions` which define
		  the priors for the activation parameters H1, S1, H2, S2
		* `tau` is the measurement period of `data`

	Returns:
		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
	"""

	thetas = theta[:3]

	# ensures that e1 < e2
	if thetas[0] > thetas[1]:
		return -np.inf

	H1, S1, H2, S2 = theta[3:]
	k1 = TST(H1,S1,T)
	k2 = TST(H2,S2,T)

	lnprior = 0

	# evaluating the priors for activation parameters
	for i in range(len(E_priors)):
		lnprior += E_priors[i].lnpdf(theta[3 + i])

	# evaluating priors for E_fret's and noise. Since the priors for the rate constants have
	# already been evaluated in terms of activation parameters, two values of rate constants
	# (which fall in the range given by the fake priors) are hard-coded here, which only add
	# constant to the posterior probability, and do not change the maximum. 
	lnprior += theta_prior.lnpdf(np.concatenate((thetas, np.array([0.5, 0.5]))))

	if np.isnan(lnprior):
		return -np.inf
	elif not np.isfinite(lnprior):
		return -np.inf

	y = lnprior

	for i in range(len(T)):
		params = np.concatenate((thetas, np.array([k1[i], k2[i]])))
		if ll_version == "CUDA":
			y += log_likelihood(params,i,tau,device=device)
		else:
			y += log_likelihood(params,data[i],tau,device=device)

	if np.isnan(y):
		return -np.inf
	else:
		return y


def setup(data, T, priors, E_priors, tau, nwalkers, initialize='rvs', device=0):
	"""
	Prepare the MCMC sampler

	Input:
		* `data` is a list of 5 1D `np.ndarray`s of the time series at 5 temperature points to analyze
		* `T` is a `np.ndarray` of 5 temperature points that `data` corresponds to
		* `priors` is a `biasd.distributions.parameter_collection` containing the prior probability
		  distributions for e1, e2, sigma (along with fake k1 and k2 priors) for the BIASD calculation
		* `E_priors` is a list of probability distributions drawn from `biasd.distributions` which define
		  the priors for the activation parameters H1, S1, H2, S2
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
	u = [0,1,2] 	#used to ignore k1 and k2 from 'biasd.distributions.parameter_collection' functions

	if isinstance(initialize,np.ndarray) and initialize.shape == (nwalkers,7):
		initial_positions = initialize

	elif initialize == 'rvs':
		H1 = E_priors[0].rvs(nwalkers).flatten()
		S1 = (H1 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S1 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		H2 = E_priors[2].rvs(nwalkers).flatten()
		S2 = (H2 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S2 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		initial_positions = np.array([np.concatenate((priors.rvs(1).flatten()[u], np.array([H1[i], S1[i], H2[i], S2[i]]))) for i in range(nwalkers)])

	elif initialize == 'mean':
		H1 = E_priors[0].mean()
		S1 = (H1 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S1 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		H2 = E_priors[2].mean()
		S2 = (H2 - 71400.)/300.		#used to maintain negative correlation between H and S in initial positions
		S2 += np.random.normal(size=nwalkers)*10 ## +/- 10 J/mol/K to pass emcee cov conditioning check
		initial_positions = np.array([np.concatenate((np.array([p.mean() for p in priors]).flatten()[u], np.array([H1,S1,H2,S2])), axis = 0) for _ in range(nwalkers)])

	else:
		raise AttributeError('Could not initialize the walkers. Try calling with initialize=\'rvs\'')

	# Slap-dash hackery to make sure the first E_fret is the lower one
	for i in range(initial_positions.shape[0]):
		if initial_positions[i,0] > initial_positions[i,1]:
			temp = initial_positions[i,0]
			initial_positions[i,0] = initial_positions[i,1]
			initial_positions[i,1] = temp

	# Loading the data on the GPU to make it persistent throughout the calculation
	if ll_version == "CUDA":
		from .likelihood import load_cuda,free_cuda
		free_cuda(device)
		load_cuda(data,device)

	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_temperature_posterior, args=[data,T,priors,E_priors,tau,device])

	return sampler,initial_positions