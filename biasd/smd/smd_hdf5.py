#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py

def _addhash(hdf5_item):
	"""
	Acts on an h5py item to add SMD identification attributes:
		* Time Created
		* SMD Hash ID
	Input:
		* `hdf5_item` is an h5py item (e.g., file, group, or dataset)
	"""
	from time import ctime
	from hashlib import md5

	time = ctime()
	hdf5_item.attrs['time created'] = time
	# h5py items don't really hash, so.... do this, instead.
	# Should be unique, and the point of the hash is for identification
	hdf5_item.attrs['SMD hash ID'] = md5(time + str(hdf5_item.id.id) + str(np.random.rand())).hexdigest()

def new(filename,force=False):
	"""
	Create and close a new HDF5 file.
	Input:
		* `filename` is the output filename to save.
		* `force` is a boolean whether to overwrite if the file already exists.
	Returns:
		* `f` is the output h5py `File`.
	"""
	# Create file and fail if it exists
	mode = 'x'
	if force:
		mode = 'w'
	try:
		f = h5py.File(filename,mode)
		_addhash(f)
		return f
	except:
		raise smd_exists

def load(filename):
	'''
	Open an HDF5 file in R/W mode with h5py
	Input:
		* `filename` is the input filename to open
	Returns:
		* an h5py file
	'''
	# Create file and fail if it exists
	try:
		f = h5py.File(filename,'r+')
	except:
		raise smd_fail(filename)
	return f

def save(f):
	'''
	Save changes made to an HDF5 file and close it.
	Input:
		* `f` is a h5py HDF5 file
	'''
	try:
		f.flush()
	except:
		raise smd_save()
	f.close()

def loadtxt(filename):
	'''
	Safely load binary (.npy), tab, or comma delimited arrays
	Input:
		* `filename` is the file to open
	Returns:
		* a numpy `ndarray`
	'''
	try:
		return np.load(filename)
	except:
		pass
	try:
		f = open(filename,'r')
		line1 = f.readline()
		f.close()
		if line1.count(','):
			return np.loadtxt(filename,delimiter=',')
		else:
			return np.loadtxt(filename)
	except:
		raise smd_fail(filename)

class smd_exists(Exception):
	""" Error if a file already exists """
	def __init__(self):
		Exception.__init__(self,"SMD I/O error: File Already Exists")

class smd_fail(Exception):
	""" Error for opening a file """
	def __init__(self,filename):
		Exception.__init__(self,"SMD I/O error: Couldn't open %s"%(filename))

class smd_malformed(Exception):
	""" Error if file isn't formed as expected """
	def __init__(self):
		Exception.__init__(self,"SMD I/O error: file is not formed as expected")

class smd_save(Exception):
	""" Error if file isn't saved """
	def __init__(self):
		Exception.__init__(self,"SMD I/O error: could not flush buffers to file ")
