.. _getstarted:

Getting Started
===============

In general, BIASD uses the SMD data format (DOI: 10.1186/s12859-014-0429-4) for data storage, though this is not required. It also uses `emcee` (arXiv:1202.3665) to perform the Markov chain Monte Carlo (MCMC), though the Laplace approximation is also provided, which does not use emcee. To start using BIASD, you will need to first install some Python packages, and maybe compile some libraries if your want reasonable computational performance.

Software Requirements
---------------------

Most of BIASD is written in Python. In general, you can run a Python script (.py) from a terminal with

.. code-block:: bash
	
	 python myscript.py
	
BIASD depends upon several Python packages, which must be installed. These can easily be obtained with the `conda` package manager, which is most efficiently obtained by installing `Continuum Analytics' Miniconda <http://conda.pydata.org/miniconda.html>`_. After installing `Miniconda` (the 64 bit, Python 2.7 version), you can then get the Python packages you will need.

You will need to use `conda` to install `pip` (version 9.0), `numpy` (version 1.11), `scipy` (verison 0.18), and `matplotlib` (version 1.5). For saving in the HDF5 SMD format, you will also need `h5py` (version 2.6) -- these versions are up-to-date as of writing. In a terminal, install these packages with 

.. code-block:: bash
	
	conda install pip numpy scipy matplotlib h5py

Once you have `pip` installed, you can install `emcee` for MCMC and `corner` for plotting purposes from a terminal window with 

.. code-block:: bash
	
	pip install emcee corner

Adding BIASD to the Python Path
-------------------------------
In order for your version of Python to find BIASD, you'll need to add the folder containing the BIASD code into your environmental variable PYTHONPATH. To do this, you'll need to edit your shell startup script with a text editor. This is located at `/Users/<username>/.profile` for mac, or `/home/<username>/.bashrc` if you're using bash for linux/unix; here, you should replace `<username>` with your username. At the end of your file, add the line

.. code-block:: bash

	export PYTHONPATH="${PYTHONPATH}:/path/to/the/biasd/folder/you/downloaded"
	
which assumes that you downloaded the BIASD package to `/path/to/the/biasd/folder/you/downloaded`. Note, that you need to either restart your terminal window, or source the startup script, e.g.

.. code-block:: bash

	source ~/.bashrc
	
After this, you should be able to use BIASD with Python. However, you should compile the likelihood function if you want better performance.