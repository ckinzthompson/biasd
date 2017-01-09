.. _code_distributions:

Distributions
=============

This page gives the details about the code in biasd.distributions.

Some standard probability distributions
+++++++++++++++++++++++++++++++++++++++

.. automodule:: distributions
	:members: beta, gamma, normal, uniform

Convert between distributions
+++++++++++++++++++++++++++++

.. autofunction:: distributions.convert_distribution

Distributions can be collected for priors or posteriors 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. autoclass:: distributions.parameter_collection
	
Collections can be visualized
+++++++++++++++++++++++++++++

.. autoclass:: distributions.viewer

Finally, you can easily generate a few useful collections using
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. automodule:: distributions
	:members: uninformative_prior,guess_prior
	