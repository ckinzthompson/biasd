"""
.. module:: likelihood

	:synopsis: Contains functions to calculate the likelihood function for BIASD

"""

import numpy as np
np.seterr(all="ignore")

from .py_numba_biasd_likelihood import log_likelihood_numba, nosum_log_likelihood_numba
from .py_numba_biasd_likelihood import log_likelihood_numba_2sigma, nosum_log_likelihood_numba_2sigma
from .py_scipy_biasd_likelihood import log_likelihood_scipy, nosum_log_likelihood_scipy

from .py_cuda_biasd_likelihood import _flag_cuda
if _flag_cuda:
	from .py_cuda_biasd_likelihood import log_likelihood_cuda, nosum_log_likelihood_cuda, load_cuda, free_cuda, cuda_device_count, test_data

from .py_c_biasd_likelihood import _flag_c
if _flag_c:
	from .py_c_biasd_likelihood import log_likelihood_c, nosum_log_likelihood_c

ll_version = ''
log_likelihood = None
nosum_log_likelihood = None

def use_python_scipy_ll():
	global ll_version, log_likelihood, nosum_log_likelihood
	ll_version = "Python"	
	log_likelihood = log_likelihood_scipy
	nosum_log_likelihood = nosum_log_likelihood_scipy
	print(f'Using {ll_version} (Scipy)')

def use_python_numba_ll():
	global ll_version, log_likelihood, nosum_log_likelihood
	ll_version = "Python"
	log_likelihood = log_likelihood_numba
	nosum_log_likelihood = nosum_log_likelihood_numba
	print(f'Using {ll_version} (Numba)')

def use_python_numba_ll_2sigma():
	global ll_version, log_likelihood, nosum_log_likelihood
	ll_version = "Python-2sigma"
	log_likelihood = log_likelihood_numba_2sigma
	nosum_log_likelihood = nosum_log_likelihood_numba_2sigma
	print(f'Using {ll_version} (Numba-2sigma)')

use_python_ll = use_python_numba_ll

if _flag_cuda:
	def use_cuda_ll():
		global ll_version, log_likelihood, nosum_log_likelihood
		ll_version = "CUDA"
		log_likelihood = log_likelihood_cuda
		nosum_log_likelihood = nosum_log_likelihood_cuda
		print(f'Using {ll_version}')
	## One time loading, and they'll never be overloaded 

else:
	def load_cuda(*args,**kw_args):
		raise Exception('No CUDA')
	def free_cuda(*args,**kw_args):
		raise Exception('No CUDA')
	def test_data(*args,**kw_args):
		raise Exception('No CUDA')

if _flag_c:
	def use_c_ll():
		global ll_version,log_likelihood, nosum_log_likelihood
		ll_version = "C"
		log_likelihood = log_likelihood_c
		nosum_log_likelihood = nosum_log_likelihood_c
		print(f'Using {ll_version}')

#### Default Choice -- numba is usually faster than C
if _flag_cuda:
	use_cuda_ll()
elif _flag_c:
	use_c_ll()
else:
	use_python_ll()


################################################################################
################################################################################


def test_speed(n,dpoints = 5000,device=0):
	"""
	Test how fast the BIASD integral runs.
	Input:
		* `n` is the number of times to repeat the test
		* `dpoints` is the number of data points in each test

	Returns:
		* The average amount of time per data point in seconds.
	"""
	import time
	d = np.linspace(-.2,1.2,dpoints)
	ts = []
	y = 0
	t00 = time.time()
	if ll_version == 'CUDA': ## persistent 
		free_cuda(device)
		load_cuda([d,],device) 
	for i in range(n):
		t0 = time.time()
		if ll_version == 'CUDA':
			y = log_likelihood(np.array([0.,1.,.05,3.,8.]),0,.1,device=device)
		else:
			y = log_likelihood(np.array([0.,1.,.05,3.,8.]),d,.1,device=device)
		t1 = time.time()
		ts.append(t1-t0)
	t11 = time.time()
	print(y)
	dt = np.median(ts)
	print("Total time for "+str(n)+" runs: ",np.around(t11-t00,4)," (s)")
	print('Average speed: ', np.around(dt/d.size*1.e6,4),' (usec/datapoint)')
	return np.around(dt/d.size*1.e6,4),y

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
	if np.isnan(y) or theta[0] >= theta[1]:
		return -np.inf
	else:
		return y