.. _compileguide:

Compiling the Likelihood
========================


Background
----------
The BIASD log-likelihood function is something like

.. math::
	
	ln(\mathcal{L}) \sim \sum\limits_t ln \left( \delta(f) + \delta(1-f) + \int\limits_0^1 df \cdot \rm{blurring} \right)
	

Unfortunately, the integral in the logarithm makes it difficult to compute. It is the rate limiting step for this calculation. Therefore, this package comes with the log-likelihood function written in python, JIT-compiled python, C, and also in CUDA. There are four versions in the ``./biasd/likelihood`` directory. You can use any of the above if compiled, or a version written in Python if you don't want to compile anything. 


How to Compile
--------------

There's a Makefile included in the package that will allow you to easily compile all of the libraries necessary to calculate BIASD likelihoods. In the terminal, move to the ``./biasd/likelihood`` directory using ``cd``, and make them with

.. code-block:: bash
	
	make
	
Some might fail, for instance if you don't have a CUDA-enabled GPU, but you'll compile as many as possible into the ``./biasd/likelihood`` directory.

Testing Speed
-------------
To get a feeling for how long it takes the various versions of the BIASD likelihood function to execute, you can use the test function in the likelihood module. For instance, try

.. code-block:: python

	import biasd as b
	
	# Switch to the Python version
	b.likelihood.use_python_ll()

	# Run the test 10 times, for 5000 datapoints
	b.likelihood.test_speed(10,5000)
	
	# Switch to the C version and test
	b.likelihood.use_C_ll()
	b.likelihood.test_speed(10,5000)
	
	# Switch to the CUDA version and test
	b.likelihood.use_CUDA_ll()
	b.likelihood.test_speed(10,5000)
	

The actual execution time depends upon the rate constants.