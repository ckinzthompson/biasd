.. _code:

Code
====

This page gives the details about the BIASD code.

Distributions
-------------

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

Likelihood
--------------------------

Nothin' here yet.

SMD
---

The Single-Molecule Dataset (SMD) format is a standardized data format for use in time-resolved, single-molecule experiments (e.g, smFRET, force spectroscopy, etc.). It was published in collaboration between the `Gonzalez <http://www.columbia.edu/cu/chemistry/groups/gonzalez/index.html>`_ and `Herschlag <http://cmgm.stanford.edu/herschlag/>`_ labs (Greenfield, M *et al*. *BMC Bioinformatics* **2015**, *16*, 3. `DOI: 10.1186/s12859-014-0429-4 <http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0429-4>`_). A github page with some `Matlab <https://smdata.github.io>`_ and `Python <https://github.com/smdata/smd-python>`_ code exists.

Work with SMD data
++++++++++++++++++
.. automodule:: smd
	:members: new, load, save

Add BIASD results to an SMD object
++++++++++++++++++++++++++++++++++
.. automodule:: smd.add
	:members:

Read BIASD results from an SMD object into a useful format
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. automodule:: smd.read
	:members:

MCMC
----

Markov chain Monte Carlo

.. automodule:: mcmc
	:members:

Laplace
-------

Laplace Approximation


Utils
---------

.. autofunction:: utils.baseline.remove_baseline