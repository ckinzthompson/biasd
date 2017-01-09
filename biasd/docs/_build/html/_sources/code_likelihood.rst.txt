.. _code_likelihood:

Likelihood
================

This page gives the details about the code in biasd.likelihood.

Switch the log-likelihood function
++++++++++++++++++++++++++++++++++++++++++++

There are two main functions that you use for BIASD in `biasd.likelihood`. One to calculate the log-likelihood function, and the other to calculate the log-posterior function, which relies on the log-likelihood function. However, in truth, there are several different version of the log-likehood function that all accept the same arguments and return the results. There's one written in Python, (two) written in C, and one written in for CUDA. Assuming that they are compiled (i.e., C or CUDA), you can toggle between them to choose which version the log-likelihood function uses. In general, you'll want to use the C version if you have only a few data points (< 500), since it is fast and it allows you to use multiple processors when performing MCMC with emcee. If you have a lot of data points, you'll probably want to use the CUDA version, where each CUDA-core calculates the log-likelihood of a single data point. Anyway, you can toggle between the versions using

.. code-block:: python
	
	import biasd as b
	
	# Switch to the slow, python implementation
	b.likelihood.use_python_ll()
	
	# Switch to the medium, parallelizable C version
	b.likelihood.use_c_ll()
	
	# Switch to the high-throughput CUDA version
	b.likelihood.use_cuda_ll()

Finally, you can test the speed per datapoint of each of these version with

.. autofunction:: likelihood.test_speed

If you're ever confused about which version you're using, you can check the `biasd.likelihood.ll_version` variable.

Warning:
	Changing `biasd.likelihood.ll_version` will not switch which likelihood function is being used.

Inference-related functions
+++++++++++++++++++++++++++

.. automodule:: likelihood
	:members: log_likelihood, log_posterior