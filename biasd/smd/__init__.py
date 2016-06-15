#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. module:: smd

	:synopsis: Allows you to use the single-molecule dataset (SMD) format, and integrate BIASD results into them.

.. moduleauthor:: Colin Kinz-Thompson <cdk2119@columbia.edu>

"""

from ._general_smd import test_smd, new, load, save
from . import add
from . import read