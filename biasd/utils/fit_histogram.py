"""
Fit histograms to the BIASD Likelihood function
------------------
	This might be useful for exploring data, or plotting
"""

import numpy as _np

def likelihood_one_at_a_time(theta,d,tau):
	"""
	Calculate the likelihood (not log-likelihood) for each point in a string of data. This might be useful for plotting, for instance:
	theta = _np.array((0.,1.,.05,3.,8.))
	x = np.linspace(-.5,1.5,1000)
	y = likelihood_one_at_a_time(theta,x,0.1)
	plt.plot(x,y)
	"""
	from biasd.likelihood import log_likelihood
	## hacks, terrible hacks, b/c I'm too lazy to rewrite the likelihood fxns to not return a sum
	lny = map(lambda x: log_likelihood(theta,x,tau),d.tolist())
	return _np.exp(lny)

def fit(data,tau,guess=None):
	"""
	Fits a histogram of data (numpy.ndarray) to the BIASD likelihood function with time period tau.
	The initial guess can be provided as:
		- a BIASD parameter_collection --> it uses the mean
		- a numpy.ndarray
		- Nothing... in which case it will try to guess
	Returns the best fitted parameters and the covariances
	"""
	from scipy.optimize import curve_fit

	from biasd.distributions import parameter_collection,guess_prior
	if isinstance(guess,parameter_collection):
		guess = guess.mean()
	elif isinstance(guess,_np.ndarray):
		guess = guess
	else:
		guess = guess_prior(data,tau=tau).mean()
		
	hy,hx = _np.histogram(data,bins=int(data.size**.5),normed=True)
	hx = .5*(hx[1:] + hx[:-1])
	
	fitted_params,covars = curve_fit(lambda x,e1,e2,sig,k1,k2: likelihood_one_at_a_time(_np.array((e1,e2,sig,k1,k2)),x,tau),hx,hy,p0=guess)
	return fitted_params,covars