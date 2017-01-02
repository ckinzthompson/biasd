# BIASD Quick Start Guide

Guide for Unix-based computers (Mac or Linux)

## 1. Documentation

The documentation on how to use BIASD (explanations, example scripts, and function documentation), is on the [Read the Docs](https://readthedocs.org) -style documentation website. This website will be found [online in the near future](http://not_hosted_anywhere_yet), but is also included locally in the BIASD git repository. The main page HTML file is located at `/path/to/the/folder/biasd/biasd/docs/_build/html/index.html`. Opening this in a web browser will allow you to browse the entire documentation. This documentation describes how to use BIASD, but also how to compile the likelihood function in C or on a GPU for faster execution, and also how to perform some analysis.

## 2. Prerequisites

In order to setup BIASD, you’ll need to be familiar with the Unix shell (e.g. terminal application on mac), and you absolutely must have the following

### Python

Both mac OS and your linux distribution probably have Python 2.7 already installed, but double check by entering

```sh
python --version
```

If you do not have python for some reason, installing conda (see below) will also install python.

### Python Packages

BIASD has several python package dependences. While there are several ways to acquire these, this is done (fairly) easily using conda. If you don’t have conda installed already, you can get a minimal version called miniconda [here](http://conda.pydata.org/miniconda.html) – install the 64-bit version for python 2.7 for your OS.

Once you’ve installed conda, you will need to use it to install _pip_ (version 9.0), _numpy_ (version 1.11), _scipy_ (verison 0.18), and _matplotlib_ (version 1.5). For saving in the HDF5 SMD format, you'll need _h5py_ (version 2.6). Other versions might work, but are up-to-date as of writing. In a Unix shell, enter

```sh
$ conda install pip numpy scipy matplotlib h5py
```

You’ll also need _emcee_ (version 2.1) and _corner_ (version 2.0) for full Markov chain Monte Carlo (MCMC) functionality. How to acquire these packages is described in the BIASD documentation, but can install them now by entering

```sh
$ pip install emcee corner
```
in a Unix shell.

### BIASD

Either get BIASD from the git repository using  
`$ git clone git@gitlab.com:ckinztho/biasd.git`  
from a Unix shell. You’ll need the folder containing the BIASD code in your path. There are three ways to do this

1.  Work (i.e. save scripts and execute them) from within the folder downloaded from github/unzipped (i.e. `/path/to/the/folder/BIASD`).
2.  Permanently add this to your path by editting your shell RC file using a text editor. On mac this will be `~/.profile`; on linux, it depends on your shell, but if you don’t know what this is about, it’s probably `~/.bashrc`. Here you’ll need to add a line: `export PATH="/path/to/the/folder/biasd:$PATH"`, where the code would be in the folder `/path/to/the/folder/biasd/biasd`. If you’ve got a Unix shell open, you’ll need either close it and open it again, or issue the command `$ source ~/.profile` or `$ source ~/.bashrc`.
3.  Add the location of the code to the path at the beginning of each python script as:

```python
import sys
sys.path.append("/path/to/the/folder/BIASD")
```

## 3. Running Python Scripts

Python can be run several different ways: interactively, through notebooks, or by executing a script.

1.  To run python interactively, use [IPython](https://ipython.org). Install it using `$ conda install ipython`, and launch the ipython shell using `$ ipython`. Now any python commands you type in that shell will be executed.
2.  To use python notebooks for documentation purposes, install [Jupyter](http://jupyter.org) using `$ conda install jupyter ipykernel`. You can start the notebook server using `$ jupyter notebook`, which will open in a web browser.
3.  To simply execute a script (written in a text editor and saved as a `.py` file), in a Unix shell, issue the command `$ python scriptnamehere.py`.