import numpy as np
import biasd as b
import pytest
 
truth = -1109.3545762573976 ## this is the scipy number; numba/C is slightly different b/c of adapative integration choice

def test_speed_scipy():
	b.likelihood.use_python_scipy_ll()
	t,y = b.likelihood.test_speed(10,1000)
	assert np.allclose(y,truth)

def test_speed_numba():
	b.likelihood.use_python_numba_ll()
	t,y = b.likelihood.test_speed(10,1000)
	assert np.allclose(y,truth)

def test_speed_c():
	b.likelihood.use_c_ll()
	t,y = b.likelihood.test_speed(10,1000)
	assert np.allclose(y,truth)

def test_speed_cuda():
	b.likelihood.use_cuda_ll()
	ndevices = b.likelihood.cuda_device_count()
	for device in range(ndevices):
		t,y = b.likelihood.test_speed(10,1000,device=device)
		assert np.allclose(y,truth)


if __name__ == '__main__':
	test_speed_scipy()
	test_speed_numba()
	test_speed_c()
	test_speed_cuda()
	