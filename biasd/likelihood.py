"""
.. module:: likelihood

	:synopsis: Contains functions to calculate the likelihood function for BIASD

"""

import ctypes as _ctypes
from sys import platform as _platform
import os as _os
import numpy as _np
from functools import reduce
_np.seterr(all="ignore")
_lib_path = _os.path.dirname(_os.path.abspath(__file__)) + "/lib/"

# Relative error for numerical integration
_eps = 1e-10

_cuda_d_pointer = None
_cuda_ll_pointer = None

###########
### The C functions are for the log-likelihood of N datapoints.
### They return a void, and as input take:
### 	int N,
### 	double * d,
### 	double * ll,
### 	double ep1,
### 	double ep2,
### 	double sigma,
### 	double k1,
### 	double k2,
### 	double tau
###
### where d is a point to the input datapoints,
### and ll is a point to the output numpy array.
##########

#Try to load CUDA log-likelihood .so
# try:
if 1:
	_sopath = _lib_path + 'biasd_cuda'
	_lib_cuda = _np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....

	_lib_cuda.log_likelihood.argtypes = [
		_ctypes.c_int,
		_ctypes.c_int,
		_ctypes.POINTER(_ctypes.c_double),
		_ctypes.POINTER(_ctypes.c_double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_np.ctypeslib.ndpointer(dtype = _np.double) ]
	_lib_cuda.log_likelihood.restype  = _ctypes.c_void_p

	_lib_cuda.sum_log_likelihood.argtypes = [
		_ctypes.c_int,
		_ctypes.c_int,
		_ctypes.POINTER(_ctypes.c_double),
		_ctypes.POINTER(_ctypes.c_double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double ]
#	_lib_cuda.log_likelihood.restype  = _ctypes.POINTER(_ctypes.c_double)
	_lib_cuda.sum_log_likelihood.restype  = _ctypes.c_double

	# _lib_cuda.device_count.argtypes = [_ctypes.c_void_p]
	# _lib_cuda.device_count.restype = _ctypes.c_int
	# _lib_cuda.cuda_errors.argtypes = [_ctypes.c_int]
	# _lib_cuda.cuda_errors.restype = _ctypes.c_int

	_lib_cuda.load_data.argtypes = [_ctypes.c_int,_ctypes.c_int,_np.ctypeslib.ndpointer(dtype = _np.double),_ctypes.POINTER(_ctypes.c_void_p),_ctypes.POINTER(_ctypes.c_void_p)]
	_lib_cuda.load_data.restype = _ctypes.c_void_p

	_lib_cuda.free_data.argtypes = [_ctypes.POINTER(_ctypes.c_void_p),_ctypes.POINTER(_ctypes.c_void_p)]
	_lib_cuda.free_data.restype = _ctypes.c_void_p

	print("Loaded CUDA Library:\n"+_sopath+".so")
	_flag_cuda = True
# except:
	# _flag_cuda = False


### Try to load C log-likelihood .so
try:
	_sopath = _lib_path + 'biasd_c'

	_lib_c = _np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....

	_lib_c.log_likelihood.argtypes = [
		_ctypes.c_int,
		_np.ctypeslib.ndpointer(dtype = _np.double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_np.ctypeslib.ndpointer(dtype = _np.double) ]
	_lib_c.log_likelihood.restype  = _ctypes.c_void_p

	_lib_c.sum_log_likelihood.argtypes = [
		_ctypes.c_int,
		_np.ctypeslib.ndpointer(dtype = _np.double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double ]
	_lib_c.sum_log_likelihood.restype  = _ctypes.c_double

	print("Loaded .C Library:\n"+_sopath+".so")
	_flag_c = True
except:
	_flag_c = False


if _flag_cuda:
	def _log_likelihood_cuda(theta,data,tau,device=0):
		"""
		Calculate the log of the BIASD likelihood function at `theta` using the data `data` given the time period of the data as `tau`.

		CUDA Version
		"""
		global _eps,_cuda_d_pointer,_cuda_ll_pointer

		epsilon = _eps
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')
		_cuda_d_pointer = None
		_cuda_ll_pointer = None

		if _cuda_d_pointer is None or _cuda_ll_pointer is None:
			_cuda_d_pointer = _ctypes.POINTER(_ctypes.c_void_p)
			_cuda_ll_pointer = _ctypes.POINTER(_ctypes.c_void_p)
			_lib_cuda.load_data(device,data.size,data,_cuda_d_pointer,_cuda_ll_pointer)

		y = _lib_cuda.sum_log_likelihood(device,data.size, _cuda_d_pointer, _cuda_ll_pointer, e1, e2, sigma, sigma, k1, k2, tau,epsilon)
		# if device >= 0:
		# 	if _lib_cuda.cuda_errors(device) == 1:
		# 		raise Exception('Cuda Error: Check Cuda code')
		return y
#		llp = _lib_cuda.log_likelihood(data.size, data, e1, e2, sigma, k1, k2, tau,epsilon)
#		return _np.ctypeslib.as_array(llp,shape=data.shape)

	def _nosum_log_likelihood_cuda(theta,data,tau,device=0):
		global _eps,_cuda_d_pointer,_cuda_ll_pointer
		epsilon = _eps
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')

		if _cuda_d_pointer is None or _cuda_ll_pointer is None:
			_cuda_d_pointer = _ctypes.POINTER(_ctypes.c_void_p)
			_cuda_ll_pointer = _ctypes.POINTER(_ctypes.c_void_p)
			_lib_cuda.load_data(device,data.size,data,_cuda_d_pointer,_cuda_ll_pointer)

		ll = _np.empty_like(data)
		_lib_cuda.log_likelihood(device,data.size, _cuda_d_pointer, _cuda_ll_pointer, e1, e2, sigma, sigma, k1, k2, tau,epsilon,ll)
		# if device >= 0:
		# 	if _lib_cuda.cuda_errors(device) == 1:
		# 		raise Exception('Cuda Error: Check Cuda code')
		return ll

	def _free_cuda():
		global _cuda_d_pointer,_cuda_ll_pointer
		_lib_cuda.free_data(_cuda_d_pointer,_cuda_ll_pointer)
		_cuda_d_pointer = None
		_cuda_ll_pointer = None

	def use_cuda_ll():
		global log_likelihood
		global ll_version
		global nosum_log_likelihood
		ll_version = "CUDA"
		log_likelihood = _log_likelihood_cuda
		nosum_log_likelihood = _nosum_log_likelihood_cuda


if _flag_c:
	def _log_likelihood_c(theta,data,tau,device=None):
		"""
		Calculate the individual values of the log of the BIASD likelihood function at :math:`\\Theta`

		Input:
			* `theta` is a `np.ndarray` of the parameters to evaluate
			* `data is a 1D `np.ndarray` of the time series to analyze
			* `tau` is the measurement period of each data point in `data`

		Returns:
			* A 1D `np.ndarray` of the log-likelihood for each data point in `data`
		"""
		global _eps
		epsilon = _eps
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')
		return _lib_c.sum_log_likelihood(data.size, data, e1, e2, sigma, sigma, k1, k2, tau,epsilon)
#		llp = _lib_c.log_likelihood(data.size, data, e1, e2, sigma, k1, k2, tau,epsilon)
#		return _np.ctypeslib.as_array(llp,shape=data.shape)

	def _nosum_log_likelihood_c(theta,data,tau,device=None):
		global _eps
		epsilon = _eps
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')
		ll = _np.empty_like(data)
		_lib_c.log_likelihood(data.size, data, e1, e2, sigma, sigma, k1, k2, tau,epsilon,ll)
		return ll


	def use_c_ll():
		global log_likelihood
		global ll_version
		global nosum_log_likelihood
		ll_version = "C"
		log_likelihood = _log_likelihood_c
		nosum_log_likelihood = _nosum_log_likelihood_c

from scipy.integrate import quad as _quad
from scipy import special as _special
def _python_integrand(x,d,e1,e2,sigma1,sigma2,k1,k2,tau):
	"""
	Integrand for BIASD likelihood function
	"""
	#Ensures proper support
	if x < 0. or x > 1. or k1 <= 0. or k2 <= 0. or sigma1 <= 0. or sigma2 <= 0. or tau <= 0. or e1 >= e2:
		return 0.
	else:
		k = k1 + k2
		p1 = k2/k
		p2 = k1 /k
		y = 2.*k*tau * _np.sqrt(p1*p2*x*(1.-x))
		z = p2*x + p1*(1.-x)
		varr = sigma1**2. * x + sigma2**2. *(1.-x)
		pf = 2.*k*tau*p1*p2*(_special.i0(y)+k*tau*(1.-z)*_special.i1(y)/y)*_np.exp(-z*k*tau)
		py = 1./_np.sqrt(2.*_np.pi*varr)*_np.exp(-.5/varr*(d-(e1*x+e2*(1.-x)))**2.) * pf
		return py

def _python_integral(d,e1,e2,sigma1,sigma2,k1,k2,tau):
	"""
	Use Gaussian quadrature to integrate the BIASD integrand across df between f = 0 ... 1
	"""
	return _quad(_python_integrand, 0.,1.,args=(d,e1,e2,sigma1,sigma2,k1,k2,tau), limit=1000)[0]
_python_integral = _np.vectorize(_python_integral)

def _p_gauss(x,mu,sigma):
	return 1./_np.sqrt(2.*_np.pi*sigma**2.) * _np.exp(-.5*((x-mu)/sigma)**2.)

def _nosum_log_likelihood_python(theta,data,tau,device=None):
	"""
	Calculate the log of the BIASD likelihood function at theta using the data data given the time period of the data as tau.

	Python Version
	"""

	e1,e2,sigma,k1,k2 = theta
	p1 = k2/(k1+k2)
	p2 = 1.-p1
	out = _python_integral(data,e1,e2,sigma,sigma,k1,k2,tau)
	peak1 = _p_gauss(data,e1,sigma)
	peak2 = _p_gauss(data,e2,sigma)
	out += p1*peak1*_np.exp(-k1*tau)
	out += p2*peak2*_np.exp(-k2*tau)

	#Don't use -infinity
	return _np.log(out)

def _log_likelihood_python(theta,data,tau,device=None):
	return _np.nansum(_nosum_log_likelihood_python(theta,data,tau))

def use_python_ll():
	global log_likelihood
	global nosum_log_likelihood
	global ll_version
	ll_version = "Python"
	try:
		from .src import log_likelihood as numba_ll
		from .src import sum_log_likelihood as numba_sum_ll
		log_likelihood = numba_sum_ll
		nosum_log_likelihood = numba_ll
		ll_version = 'Python - Numba'
	except:
		log_likelihood = _log_likelihood_python
		nosum_log_likelihood = _nosum_log_likelihood_python

def test_speed(n,dpoints = 5000,device=0):
	"""
	Test how fast the BIASD integral runs.

	Input:
		* `n` is the number of times to repeat the test
		* `dpoints` is the number of data points in each test

	Returns:
		* The average amount of time per data point in seconds.

	"""
	from time import time
	d = _np.linspace(-.2,1.2,dpoints)
	t0 = time()
	for i in range(n):
		# quad(integrand,0.,1.,args=(.1,0.,1.,.05,3.,8.,.1))[0]
		y = log_likelihood(_np.array([0.,1.,.05,3.,8.]),d,.1,device=device)
	t1 = time()
	print("Total time for "+str(n)+" runs: ",_np.around(t1-t0,4)," (s)")
	print('Average speed: ', _np.around((t1-t0)/n/d.size*1.e6,4),' (usec/datapoint)')
	return _np.around((t1-t0)/n/d.size*1.e6,4)


### Default to Python implementation
log_likelihood = _log_likelihood_python
nosum_log_likelihood = _nosum_log_likelihood_python
if _flag_cuda:
	print("Using CUDA log-likelihood")
	use_cuda_ll()
elif _flag_c:
	print("Using C log-likelihood")
	use_c_ll()
else:
	print("Defaulted to native Python log-likelihood")


#def mixture_log_likelihood(theta,data,tau):
#	n = (theta.size+1)/6
#	thetas = theta[:5*n].reshape((n,5))
#	qs = theta[5*n:]
#	qs = _np.append(qs,1.-qs.sum())
#
#	lls = _np.empty((n,data.size))
#	for i in range(n):
#		lls[i] = log_likelihood(thetas[i],data,tau)
#	ll = _np.log(_np.sum(qs[:,None]*_np.exp(lls),axis=0))
#	return ll
#


#def mixture_log_posterior(theta,data,theta_priors,population_prior,tau):
#	"""
#	Calculate the log-posterior probability for a mixture of `M` sub-systems at :math:`\\vec{ \\Theta}`
#
#	Input:
#		* `theta` is a vector of BIASD parameters and the population fractions. e.g. :math:`[ \\Theta_1, ..., \\Theta_M, f_1, ..., f_{M-1}`
#		* `data` is a 1D `np.ndarray` of the time series to analyze
#		* `theta_priors` is a list of `biasd.distributions.parameter_collection`s containing the prior probability distributions for each sub-system
#		* `population_prior` is a function that takes a 1D `np.ndarray` vector of length M containing the occupation probability of each of the `M` states, and returns the log probability prior of that point as a float.
#		* `tau` is the measurement period of `data`
#
#	Returns:
#		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
#	"""

#	n = (theta.size+1)/6

#	thetas = theta[:5*n].reshape((n,5))
#	qs = theta[5*n:]
#	qs = _np.append(qs,1.-qs.sum())
#
#
#	lnprior = 0.
#	for i in xrange(n):
#		lnprior += theta_priors[i].lnpdf(thetas[i])
#	lnprior += population_prior.lnpdf(qs)
#
#	if _np.isnan(lnprior):
#		return -_np.inf
#	elif not _np.isfinite(lnprior):
#		return -_np.inf
#	else:
#		lls = _np.empty((n,data.size))
#		for i in range(n):
#			lls[i] = log_likelihood(thetas[i],data,tau)
#		ll = _np.sum(_np.log(_np.sum(qs[:,None]*_np.exp(lls),axis=0)))
#		y = lnprior + ll
#		if _np.isnan(y):
#			return -_np.inf
#		else:
#			return y

def log_posterior(theta,data,prior_dists,tau,device=0):
	"""
	Calculate the log-posterior probability distribution at :math:`\\Theta`

	Input:
		* `theta` is a vector of the parameters (i.e., :math:`\\theta`) where to evaluate the log-posterior
		* `data` is a 1D `np.ndarray` of the time series to analyze
		* `prior_dists` is a `biasd.distributions.parameter_collection` containing the prior probability distributions for the BIASD calculation
		* `tau` is the measurement period of `data`

	Returns:
		* The summed log posterior probability distribution, :math:`p(\\Theta \\vert data) \\propto p(data \\vert \\Theta) \cdot p(\\Theta)`
	"""
	lprior = prior_dists.lnpdf(theta)
	ll = log_likelihood(theta,data,tau,device=device)
	y = lprior + ll

	# keep e1 < e2...
	if _np.isnan(y) or theta[0] >= theta[1]:
		return -_np.inf
	else:
		return y


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
	from scipy.optimize import curve_fit
	from .distributions import guess_prior

	if isinstance(guess,_np.ndarray):
		guess = guess
	else:
		guess = guess_prior(data,tau=tau).mean()

	hy,hx = _np.histogram(data,bins=int(data.size**.5),normed=True)
	hx = .5*(hx[1:] + hx[:-1])

	fitted_params,covars = curve_fit(lambda x,e1,e2,sig,k1,k2: _np.exp(nosum_log_likelihood(_np.array((e1,e2,sig,k1,k2)),x,tau,device=device)),hx,hy,p0=guess)
	return fitted_params,covars

def predictive_from_samples(x,samples,tau,device=0):
	'''
	Returns the posterior predictive distribution calculated from samples -- the average value of the likelihood function evaluated at `x` marginalized from the samples of BIASD parameters given in `samples`.

	Samples can be generated from a posterior probability distribution. For instance, after a Laplace approximation, just draw random variates from the multivariate-normal distribution -- i.e., given results in `r`, try `samples = np.random.multivariate_normal(r.mu,r.covar,100)`. Alternatively, some posteriors might already have samples (e.g., from MCMC).

	Input:
		* `x` a `np.ndarry` where to evaluate the likelihood at (e.g., [-.2 ... 1.2] for FRET)
		* `samples` is a (N,5) `np.ndarray` containing `N` samples of BIASD parameters (e.g. \Theta)
		* `tau` the time period with which to evaluate the likelihood function
	Returns:
		* `y` a `np.ndarray` the same size as `x` containing the marginalized likelihood function evaluated at x
	'''
	n = samples.shape[0]
	y = reduce(lambda x,y: x+y, [_np.exp(nosum_log_likelihood(samples[i],x,tau,device=device)) for i in range(n)])/n
	return y
