.. _code_smd:

SMD
===

This page gives the details about the code in biasd.io.

The Single-Molecule Dataset (SMD) format is a standardized data format for use in time-resolved, single-molecule experiments (e.g, smFRET, force spectroscopy, etc.). It was published in collaboration between the `Gonzalez <http://www.columbia.edu/cu/chemistry/groups/gonzalez/index.html>`_ and `Herschlag <http://cmgm.stanford.edu/herschlag/>`_ labs (Greenfield, M *et al*. *BMC Bioinformatics* **2015**, *16*, 3. `DOI: 10.1186/s12859-014-0429-4 <http://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-014-0429-4>`_). Here we have adapted the SMD data format for use with the `HDF5 <https://support.hdfgroup.org/HDF5/>`_ data format, which we access with the python library `h5py <http://www.h5py.org>`_. 

Work with SMD data
++++++++++++++++++
.. automodule:: smd
	:members: new, load, loadtxt, add_trajectories

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
	cy3 = b.smd.loadtxt('my_cy3_data.dat')
	cy5 = b.smd.loadtxt('my_cy5_data.dat')
	fret = d1/(d1+d2)

	# Create the time axis, assuming data is in (N,T) array
	tau = 0.1
	t = np.arange(fret.shape[1])*tau
	

	# Make a new HDF5 SMD file
	filename = '20161230_fret_experiment_1.hdf5'
	f = b.smd.new(filename)
	
	# Add the FRET trajectories
	b.smd.add.trajectories(f, t, fret, x_label='Time (s)', y_label='E_{FRET}')
	nmolecules  = f.attrs['number of trajectories']
	
	# Save and close the output file
	b.smd.save(f)
	
	# Setup the priors
	e1 = b.distributions.beta(1.,9.)
	e2 = b.distributions.beta(9.,1.)
	sigma = b.distributions.gamma(2.,2./.05)
	k1 = b.distributions.gamma(20.,20./3.)
	k2 = b.distributions.gamma(20.,20./8.)
	priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)
	
	# Open the output file
	f = b.smd.open(filename)
	
	# Loop over all the molecules
	for i in range(nmolecules):
		
		# Open the file again, because saving closes it
		f = b.smd.load(filename)
		
		# Get the ith trajectory reference
		trajectory = f['trajectory %d'%(i)]
		
		# Get the entire (i.e. [:] at the end) E_{FRET} data vector 
		data = trajectory['data/E_{FRET}'][:]
		# Or equivalently...
		data = trajectory['data/E_{FRET}'].value
		
		# Create a new group for this particular analysis
		group = trajectory.create_group("20161231 Laplace Analysis")
		
		# Add the value of tau to the file for reference
		group.attrs['tau'] = tau
		
		# Add the priors used to the file for reference
		b.smd.add.parameter_collection(group,priors,label="Priors")
		
		# Do some calculation on data... for instance,
		laplace_result = b.laplace.laplace_approximation(data,priors,tau)
		
		# Add the results to the file
		b.smd.add.laplace_posterior(group,laplace_result)
		
		# Save the results as we go in case of a crash
		b.smd.save(f)

Load, and read from example above

.. code-block:: python

	import biasd as b
	
	# Load the SMD
	filename ='20161230_fret_experiment_1.hdf5'
	f = b.smd.load(filename)
	
	# Read the priors for the first molecule (0)
	analysis = f['trajectory 0/20161231 Laplace Analysis']
	priors = b.smd.read.parameter_collection(analysis['Priors'])
	
	# Read the Laplace posterior object for the first molecule (0)
	lp = b.smd.read.laplace_posterior(analysis['result'])
	
	# Or if there are also MCMC results called 'result' in a group called '20170101 MCMC Analysis' in trajectory 1... 
	analysis = f['trajectory 1/20170101 MCMC Analysis']
	mcmc_results = b.smd.read.mcmc(analysis['result'])

