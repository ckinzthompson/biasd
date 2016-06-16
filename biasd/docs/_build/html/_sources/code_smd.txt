.. _code_smd:

SMD
===

This page gives the details about the code in biasd.smd.

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
	
	
Examples
++++++++

Create, add, and save example

.. code-block:: python

	import biasd as b
	import numpy as np
	
	# Load the data
	tau = 0.1
	t = np.linspace(0,1000,tau)
	cy3 = np.loadtxt('my_cy3_data.dat')
	cy5 = np.loadtxt('my_cy5_data.dat')
	fret = d1/(d1+d2)
	
	# Structure the data into NxDxT
	d = np.hstack((cy3,cy5,fret))

	# Make a new SMD
	data = b.smd.new(t,d,['Cy3','Cy5','FRET'])

	# Setup the priors
	e1 = b.distributions.beta(1.,9.)
	e2 = b.distributions.beta(9.,1.)
	sigma = b.distributions.gamma(2.,2./.05)
	k1 = b.distributions.gamma(20.,20./3.)
	k2 = b.distributions.gamma(20.,20./8.)
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)
	
	# Loop over all the molecules
	for i in range(data.attr.n_traces):
		
		# Add the priors, and Laplace posterior results
		data = b.smd.add_priors(data,i,priors)
		
		# Do a long calculations and add those results...
		
		# Save the results as we go in case of a crash
		b.smd.save('data.smd',data)

Load, and read example

.. code-block:: python

	import biasd as b
	
	# Load the SMD
	data = b.smd.load('data.smd')
	
	# Read the priors for the first molecule (0)
	priors = b.smd.read_priors(data,0)
	
	# Read the Laplace posterior object for the first molecule (0)
	lp = b.smd.read_laplace_posterior(data,0)
	
	# Read the MCMC rsults for the first molecule (0)
	mcmc_results = b.smd.read_mcmc(data,0)

