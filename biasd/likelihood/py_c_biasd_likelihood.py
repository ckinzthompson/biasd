import ctypes
import os
import numpy as np
np.seterr(all="ignore")

_lib_path = os.path.dirname(os.path.abspath(__file__))
# print(_lib_path)

_cppd = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
eps = 1e-10 # Relative error for numerical integration

_cuda_d_pointer = None
_cuda_ll_pointer = None

### Try to load C log-likelihood .so
try:
	_sopath = os.path.join(_lib_path,'c_biasd.so')
	_lib_c = np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....

	_lib_c.log_likelihood.argtypes = [
		ctypes.c_int,
		np.ctypeslib.ndpointer(dtype = np.double),
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		np.ctypeslib.ndpointer(dtype = np.double)
	]
	_lib_c.log_likelihood.restype  = ctypes.c_void_p

	_lib_c.sum_log_likelihood.argtypes = [
		ctypes.c_int,
		np.ctypeslib.ndpointer(dtype = np.double),
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double
	]
	_lib_c.sum_log_likelihood.restype  = ctypes.c_double

	print(f"Loaded .C Library: {_sopath}")

	def log_likelihood_c(theta,data,tau,device=None,epsilon=eps):
		"""
		Calculate the individual values of the log of the BIASD likelihood function at :math:`\\Theta`

		Input:
			* `theta` is a `np.ndarray` of the parameters to evaluate
			* `data is a 1D `np.ndarray` of the time series to analyze
			* `tau` is the measurement period of each data point in `data`

		Returns:
			* A 1D `np.ndarray` of the log-likelihood for each data point in `data`
		"""
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,np.ndarray):
			data = np.array(data,dtype='double')
		return _lib_c.sum_log_likelihood(data.size, data, e1, e2, sigma, sigma, k1, k2, tau,epsilon)

	def nosum_log_likelihood_c(theta,data,tau,device=None,epsilon=eps):
		e1,e2,sigma,k1,k2 = theta
		if not isinstance(data,np.ndarray):
			data = np.array(data,dtype='double')
		ll = np.empty_like(data)
		_lib_c.log_likelihood(data.size, data, e1, e2, sigma, sigma, k1, k2, tau,epsilon,ll)
		return ll

	_flag_c = True
except:
	_flag_c = False

