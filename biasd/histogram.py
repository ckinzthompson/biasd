import numpy as np
from scipy.optimize import curve_fit
from .distributions import guess_prior
from .likelihood import nosum_log_likelihood

def fit_histogram(data,tau,guess=None,device=0):
	"""
	Fits a histogram of to the BIASD likelihood function.

	Input:

		* `data` is a `np.ndarray`
		* `tau` is the measurement period
		* `guess` is an initial guess. This can be provided as:
			- a `biasd.distributions.parameter_collection`, it will use the mean
			- a `np.ndarray`
			- `Nothing...`, in which case it will try to guess

	Returns:
		* the best-fit parameters, and the covariances
	"""


	if isinstance(guess,np.ndarray):
		guess = guess
	else:
		guess = guess_prior(data,tau=tau).mean()

	hy,hx = np.histogram(data,bins=int(data.size**.5),density=True)
	hx = .5*(hx[1:] + hx[:-1])

	if hx.size <= 10:
		print('Not enough datapoints for a good histogram')
		return guess,np.zeros((guess.size,guess.size))+np.inf

	fitted_params,covars = curve_fit(lambda x,e1,e2,sig,k1,k2: np.exp(nosum_log_likelihood(np.array((e1,e2,sig,k1,k2)),x,tau,device=device)),hx,hy,p0=guess,maxfev=10000)
	return fitted_params,covars