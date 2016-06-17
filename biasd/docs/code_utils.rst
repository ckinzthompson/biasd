.. _code_utils:

Utils
====

This page gives the details about the code in biasd.utils.

Baseline Correction
-------------------

Code to correct for white-noise baseline drift. This type of drift will severely hinder the ability of BIASD to provide reasonable results.

.. automodule:: utils.baseline
	:members: remove_baseline

Example:

.. code-block:: python

	import matplotlib.pyplot as plt
	import biasd as b

	# Load some data
	data = b.smd.load('test.smd')
	d = data.data[0].values.FRET

	# Calculate some fake baseline-drift and apply it to the data
	baseline = b.utils.baseline.simulate_diffusion(d.size,1e-2)
	dd = d + baseline

	# Solve for the baseline
	baseline_results = b.utils.baseline.remove_baseline(dd)

	# Plot the results
	f,a = plt.subplots(2,sharex=True)
	a[0].plot(dd,color='b')
	a[0].plot(baseline_results.baseline,color='r')
	a[1].plot(dd - baseline_results.baseline,'k')
	plt.show()

	# Add the results to the SMD and save
	data = b.smd.add.baseline(data,0,baseline_results)
	b.smd.save('test.smd',data)


Fit histograms to the BIASD likelihood function
-----------------------------------------------
	This might be useful for exploring data, or plotting.

.. automodule:: utils.fit_histogram
	:members:
	

Clustering data
---------------

Here are some helper functions to perform clustering

.. automodule:: utils.clustering
	:members: