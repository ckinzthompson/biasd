.. _compileguide:

Compiling the Likelihood
========================


Background
----------
The BIASD log-likelihood function is something like

.. math::
	
	ln(\mathcal{L}) \sim \sum\limits_t ln \left( \delta(f) + \delta(1-f) + \int\limits_0^1 df \cdot \rm{blurring} \right)
	

Unfortunately, the integral in the logarithm makes it difficult to compute. It is the rate limiting step for this calculation, which is quite slow in Python. Therefore, this package comes with the log-likelihood function written in  C, and also in CUDA. There are three versions in the ``./src`` directory. One is in pure C -- it should be fairly straight forward to compile. The second is written in C with the `GNU Science Library (GSL) <https://www.gnu.org/software/gsl/>`_ -- it's slightly faster, but requires having installed GSL. The third is in CUDA, which allows the calculations to be performed on NVIDIA GPUs. You can use any of the above if compiled, or a version written in Python if you don't want to compile anything. 


How to Compile
--------------

There's a Makefile included in the package that will allow you to easily compile all of the libraries necessary to calculate BIASD likelihoods. First, to download GSL, go to their `FTP site <ftp://ftp.gnu.org/gnu/gsl/>`_ and download the latest version. Un-pack it, then in the terminal, navigate to the directory using ``cd`` and type 

.. code-block:: bash

	./configure
	make
	make install


Now, even if you didn't install GSL, you can compile the BIASD likelihood functions. In the terminal, move to the BIASD directory using ``cd``, and make them with

.. code-block:: bash
	
	make
	

Some might fail, for instance if you don't have a CUDA-enabled GPU, but you'll compile as many as possible into the ``./lib`` directory.

Testing Speed
-------------
To get a feeling for how long it takes the various versions of the BIASD likelihood function to execute, you can use the test function in the likelihood module. For instance, try

.. code-block:: python

	import biasd
	
	# Switch to the Python version
	biasd.likelihood.use_python_ll()

	# Run the test 10 times, for 5000 datapoints
	biasd.likelihood.test_speed(10,5000)
	
	# Switch to the C version and test
	# Note: will default to GSL over pure C
	biasd.likelihood.use_C_ll()
	biasd.likelihood.test_speed(10,5000)
	
	# Switch to the CUDA version and test
	biasd.likelihood.use_CUDA_ll()
	biasd.likelihood.test_speed(10,5000)
	

The actual execution time depends upon the rate constants, but Python is ~ 1 ms, C with GSL is around ~40 us, and CUDA (when you have many datapoints) is ~ 5 us.