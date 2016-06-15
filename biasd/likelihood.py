import ctypes as _ctypes
from sys import platform as _platform
import os as _os
import numpy as _np
_np.seterr(all="ignore")
_lib_path = _os.path.dirname(_os.path.abspath(__file__)) + "/lib/"


###########
### The C functions are for the log-likelihood of N datapoints.
### They return a void, and as input take:
### 	int N,
### 	double * d,
### 	double ep1,
### 	double ep2,
### 	double sigma, 
### 	double k1,
### 	double k2,
### 	double tau,
### 	double * ll
### where d is a point to the input datapoints,
### and ll is a point to the output numpy array.
##########

#Try to load CUDA log-likelihood .so
try:
	if _platform == 'darwin':
		_sopath = _lib_path+'biasd_ll_cuda'
	elif _platform == 'linux' or _platform == 'linux2':
		_sopath = _lib_path + 'biasd_ll_cuda'
	_lib_cuda = _np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....
	_lib_cuda.log_likelihood.argtypes = [
		_ctypes.c_int,
		_np.ctypeslib.ndpointer(dtype = _np.double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double ]
	_lib_cuda.log_likelihood.restype  = _ctypes.c_double
	print "Loaded CUDA Library:\n"+_sopath+".so"
	_flag_cuda = True
except:
	_flag_cuda = False

### Try to load C log-likelihood .so
try:
	if _os.path.isfile(_lib_path+'biasd_ll_gsl.so'):
		_sopath = _lib_path+'biasd_ll_gsl' 
	else:
		_sopath = _lib_path + 'biasd_ll' # Alternative is biasd_ll
	_lib_c = _np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....
	_lib_c.log_likelihood.argtypes = [
		_ctypes.c_int,
		_np.ctypeslib.ndpointer(dtype = _np.double),
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double,
		_ctypes.c_double ]#,
		# _np.ctypeslib.ndpointer(dtype = _np.double)
	# ]
	# _lib_c.log_likelihood.restype  = _ctypes.c_void_p
	_lib_c.log_likelihood.restype  = _ctypes.c_double
	print "Loaded .C Library:\n"+_sopath+".so"
	_flag_c = True
except:
	_flag_c = False


if _flag_cuda:
	def _log_likelihood_cuda(theta,data,tau):
		"""
		Calculate the log of the BIASD likelihood function at theta using the data data given the time period of the data as tau.
		
		CUDA Version
		"""

		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')
		return _lib_cuda.log_likelihood(data.size, data, e1, e2, sigma, k1, k2, tau)
	
	def use_cuda_ll():
		global log_likelihood
		global ll_version
		ll_version = "CUDA"
		log_likelihood = _log_likelihood_cuda
	

if _flag_c:
	def _log_likelihood_c(theta,data,tau):
		"""
		Calculate the log of the BIASD likelihood function at theta using the data data given the time period of the data as tau.
		
		C Version
		"""
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,_np.ndarray):
			data = _np.array(data,dtype='double')
		# ll = _np.empty_like(data).astype('double')
		# _lib_c.log_likelihood(data.size, data, e1, e2, sigma, k1, k2, tau, ll)
		return _lib_c.log_likelihood(data.size, data, e1, e2, sigma, k1, k2, tau)
		# return ll
	def use_c_ll():
		global log_likelihood
		global ll_version
		ll_version = "C"
		log_likelihood = _log_likelihood_c
	

from scipy.integrate import quad as _quad
from scipy import special as _special
def _python_integrand(x,d,e1,e2,sigma,k1,k2,tau):
	"""
	Integrand for BIASD likelihood function
	"""
	#Ensures proper support
	if x < 0. or x > 1. or k1 <= 0. or k2 <= 0. or sigma <= 0. or tau <= 0. or e1 >= e2:
		return 0.
	else:
		k = k1 + k2
		p1 = k2/k
		p2 = k1 /k
		y = 2.*k*tau * _np.sqrt(p1*p2*x*(1.-x))
		z = p2*x + p1*(1.-x)
		pf = 2.*k*tau*p1*p2*(_special.i0(y)+k*tau*(1.-z)*_special.i1(y)/y)*_np.exp(-z*k*tau)
		py = 1./_np.sqrt(2.*_np.pi*sigma**2.)*_np.exp(-.5/sigma/sigma*(d-(e1*x+e2*(1.-x)))**2.) * pf
		return py
		
def _python_integral(d,e1,e2,sigma,k1,k2,tau):
	"""
	Use Gaussian quadrature to integrate the BIASD integrand across df between f = 0 ... 1
	"""
	return _quad(_python_integrand, 0.,1.,args=(d,e1,e2,sigma,k1,k2,tau), limit=1000)[0]
_python_integral = _np.vectorize(_python_integral)

def _p_gauss(x,mu,sigma):
	return 1./_np.sqrt(2.*_np.pi*sigma**2.) * _np.exp(-.5*((x-mu)/sigma)**2.)

def _log_likelihood_python(theta,data,tau):
	"""
	Calculate the log of the BIASD likelihood function at theta using the data data given the time period of the data as tau.

	Python Version
	"""

	e1,e2,sigma,k1,k2 = theta
	p1 = k2/(k1+k2)
	p2 = 1.-p1
	out = _python_integral(data,e1,e2,sigma,k1,k2,tau)
	peak1 = _p_gauss(data,e1,sigma)
	peak2 = _p_gauss(data,e2,sigma)
	out += p1*peak1*_np.exp(-k1*tau)
	out += p2*peak2*_np.exp(-k2*tau)

	#Don't use -infinity
	return _np.log(out).sum()

def use_python_ll():
	global log_likelihood
	global ll_version
	ll_version = "Python"
	log_likelihood = _log_likelihood_python
	
	
def test_speed(n,dpoints = 5000):
	"""
	Test how fast the BIASD integral (python-based or C-based) runs. C-based should be ~30 us. Python-based is ~30x that.
	"""
	from time import time
	d = _np.linspace(-2,1.2,dpoints)
	t0 = time()
	for i in range(n):
		# quad(integrand,0.,1.,args=(.1,0.,1.,.05,3.,8.,.1))[0]
		y = log_likelihood([0.,1.,.05,3.,8.],d,.1)
	t1 = time()
	print "Total time for "+str(n)+" runs: ",_np.around(t1-t0,4)," (s)"
	print 'Average speed: ', _np.around((t1-t0)/n/d.size*1.e6,4),' (usec/datapoint)'
	return _np.around((t1-t0)/n/d.size*1.e6,4)
		
	
### Default to Python implementation
log_likelihood = _log_likelihood_python
if _flag_cuda:
	print "Using CUDA log-likelihood"
	use_cuda_ll()
elif _flag_c:
	print "Using C log-likelihood"
	use_c_ll()
else:
	print "Defaulted to native Python log-likelihood"


def log_posterior(params,data,prior_dists,tau):
	lprior = prior_dists.lnpdf(params)
	ll = log_likelihood(params,data,tau)
	y = lprior + ll
	
	if _np.isnan(y):
		return -_np.inf
	else:
		return y