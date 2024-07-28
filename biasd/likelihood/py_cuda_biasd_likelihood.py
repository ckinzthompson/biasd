import ctypes
import os
import numpy as np
np.seterr(all="ignore")

_lib_path = os.path.dirname(os.path.abspath(__file__))
# print(_lib_path)

_cppd = ctypes.POINTER(ctypes.POINTER(ctypes.c_double))
eps = 1e-10 # Relative error for numerical integration

data_size = []
cuda_d_pointer = []
cuda_ll_pointer = []

#Try to load CUDA log-likelihood .so
try:
	_sopath = os.path.join(_lib_path,'cuda_biasd.so')
	_lib_cuda = np.ctypeslib.load_library(_sopath, '.') ## future-self: the library has to end in .so ....

	_lib_cuda.log_likelihood.argtypes = [
		ctypes.c_int,
		ctypes.c_int,
		_cppd,
		_cppd,
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
	_lib_cuda.log_likelihood.restype  = ctypes.c_void_p

	_lib_cuda.sum_log_likelihood.argtypes = [
		ctypes.c_int,
		ctypes.c_int,
		_cppd,
		_cppd,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double,
		ctypes.c_double
	]
	_lib_cuda.sum_log_likelihood.restype  = ctypes.c_double

	_lib_cuda.load_data.argtypes = [ctypes.c_int, ctypes.c_int, np.ctypeslib.ndpointer(dtype = np.double), _cppd, _cppd]
	_lib_cuda.load_data.restype = ctypes.c_void_p

	_lib_cuda.free_data.argtypes = [ctypes.c_int, _cppd, _cppd]
	_lib_cuda.free_data.restype = ctypes.c_void_p

	_lib_cuda.device_count.argtypes = []
	_lib_cuda.device_count.restype = ctypes.c_int

	_lib_cuda.test_data.argtypes = [ctypes.c_int, ctypes.c_int, _cppd]
	_lib_cuda.test_data.restype = ctypes.c_double

	print("Loaded CUDA Library:\n"+_sopath+".so")

	def load_cuda(data, device=0):
		global cuda_d_pointer, cuda_ll_pointer, data_size

		cuda_d_pointer = []
		cuda_ll_pointer = []
		data_size = []

		for i in range(len(data)):
			cuda_d_pointer.append(ctypes.pointer(ctypes.pointer(ctypes.c_double())))
			cuda_ll_pointer.append(ctypes.pointer(ctypes.pointer(ctypes.c_double())))
			_lib_cuda.load_data(device, data[i].size, data[i], cuda_d_pointer[i], cuda_ll_pointer[i])
			data_size.append(data[i].size)

	def test_data(n=10, index=0, device=0):
		global cuda_d_pointer
		print(_lib_cuda.test_data(device, n, cuda_d_pointer[index]))

	def free_cuda(device=0):
		global cuda_d_pointer, cuda_ll_pointer, data_size

		for i in range(len(cuda_d_pointer)):
			_lib_cuda.free_data(device, cuda_d_pointer[i], cuda_ll_pointer[i])

		cuda_d_pointer = []
		cuda_ll_pointer = []
		data_size = []

	def cuda_device_count():
		return _lib_cuda.device_count()

	def log_likelihood_cuda(theta, data_or_index, tau, device=0, epsilon=eps):
		'''
		data_or_index is a cheat.
		if it's an np.ndarray, then it gets loaded onto the GPU and run
		if it's an int, then this assumes you have already loaded it (and others) onto the GPU and the index is specifying which dataset in (cuda_d_pointer...)
		'''
		global cuda_d_pointer, cuda_ll_pointer, data_size

		if type(data_or_index) is int: ## traces are preloaded, data is an index
			if data_or_index >= len(data_size): ## out of bounds
				raise Exception('Index (data) out of bounds')
			index = data_or_index
			
		elif type(data_or_index) is list: ## then like... what are you supposed to do here! which one do you want to run ? 
			raise Exception('You need to preload the data, and then give an index to run')

		elif type(data_or_index) is np.ndarray:  ## Load new data
			free_cuda(device) ## for safety
			load_cuda([data_or_index,], device) ## will be fine if it's not a list
			index = 0

		e1,e2,sigma,k1,k2 = theta
		# device = _lib_cuda.device_count() - 1

		y = _lib_cuda.sum_log_likelihood(device, data_size[index], cuda_d_pointer[index], cuda_ll_pointer[index], e1, e2, sigma, sigma, k1, k2, tau, epsilon)

		return y

	def nosum_log_likelihood_cuda(theta, data_or_index, tau, device=0, epsilon=eps):
		'''
		data_or_index is a cheat.
		if it's an np.ndarray, then it gets loaded onto the GPU and run
		if it's an int, then this assumes you have already loaded it (and others) onto the GPU and the index is specifying which dataset in (cuda_d_pointer...)
		'''
		global cuda_d_pointer, cuda_ll_pointer, data_size

		if type(data_or_index) is int: ## traces are preloaded, data is an index
			if data_or_index > len(data_size): ## out of bounds
				raise Exception('Index (data) out of bounds')
			index = data_or_index
			
		elif type(data_or_index) is list: ## then... what are you supposed to do here!? user must pick one to run
			raise Exception('You need to preload the data, and then give an index to run')

		elif type(data_or_index) is np.ndarray: 
			free_cuda(device) ## for safety
			load_cuda([data_or_index,],device) 
			index = 0
		
		e1,e2,sigma,k1,k2 = theta

		ll = np.empty(data_size[index], dtype='double')
		_lib_cuda.log_likelihood(device, data_size[index], cuda_d_pointer[index], cuda_ll_pointer[index], e1, e2, sigma, sigma, k1, k2, tau,epsilon,ll)

		return ll

	_flag_cuda = True

except:
	_flag_cuda = False
