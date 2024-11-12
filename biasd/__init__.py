"""
BIASD - Bayesian Inference for the Analysis of Sub-temporal resolution Data
"""

__version__ = "0.2.3"
__description__ = "BIASD - Bayesian Inference for the Analysis of Sub-temporal resolution Data."
__license__ = "MIT"
__url__ = "https://github.com/ckinzthompson/biasd"
__author__ = "Colin Kinz-Thompson"

from . import likelihood, distributions, mcmc
from . import laplace, histogram
from . import plot

## Variants
from . import temperature, titration, constantepsilon, constantepsilonsigma, constantepsilonsigmadead
