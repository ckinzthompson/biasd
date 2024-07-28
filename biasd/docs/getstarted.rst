.. _getstarted:

Getting Started
===============

BIASD uses `emcee` (arXiv:1202.3665) to perform the Markov chain Monte Carlo (MCMC), though the Laplace approximation is also provided, which does not use emcee. To start using BIASD, you will need to first install some Python packages, and maybe compile some libraries if your want reasonable computational performance.

Software Requirements
---------------------

Most of BIASD is written in Python. In general, you can run a Python script (.py) from a terminal with

.. code-block:: bash
	
	python myscript.py
	
BIASD depends upon several Python packages, which must be installed. These can easily be obtained with the `conda` package manager, which is most efficiently obtained by installing `Continuum Analytics' Miniconda <http://conda.pydata.org/miniconda.html>`_. After installing `Miniconda` (the 64 bit, Python 2.7 version), you can then get the Python packages you will need. In general, it is useful to have a separate environment for biasd, and to launch it everytime. To do that you can create the environment with

.. code-block:: bash
	
	conda create -n biasd python==3.9
	
and everytime you want to use the environment, type

.. code-block:: bash
	
	conda activate biasd

Once you have that prepared, you'll need to install biasd and all the libraries it needs. Just run:

.. code-block:: bash
	
	pip install git+https://github.com/ckinzthompson/biasd.git

If you want to do any testing, it's easier to have have a modifiable library and install the testing requirements

.. code-block:: bash

	git clone https://github.com/ckinzthompson/biasd.git  
	cd biasd
	pip install -e ./
	pip install -e ".[test]"

Finally, to build the documentation, navigate to the `./biasd/docs`, and run `make html`.
