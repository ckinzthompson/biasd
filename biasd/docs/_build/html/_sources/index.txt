Bayesian Inference for the Analysis of Sub-temporal-resolution Data
===================================================================

BIASD
-----

BIASD allows you to analyze Markovian signal versus time series, such as those collected in single-molecule biophysics experiments, even when the kinetics of the underlying Markov chain are faster than the signal acquisition rate. The code here has been written in python for easy implementation, but unfortunately, the likelihood function is computationally expensive since it involves a numerical integral. Therefore, the likelihood function is also provided as C code and also in CUDA with python wrappers to use them with the rest of the code base.

Contents:
=========
.. toctree::
	:maxdepth: 2
	
	getstarted
	compileguide
	examples
	gui

Code Documentation:
===================
.. toctree::
	:maxdepth: 2
	
	code_distributions
	code_laplace
	code_likelihood
	code_mcmc
	code_smd
	code_utils
	

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

