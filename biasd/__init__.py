#!/usr/bin/env python
# -*- coding: utf-8 -*-

import likelihood
import laplace
import distributions
try:
	import mcmc
except:
	print "Could not import MCMC - install emcee and corner?"
import smd
try:
	import gui
except:
	print 'Could not import gui - install PyQt5?'

__version__ = "0.1.1"
