"""
Read a BIASD result from an SMD file into a useful format. This includes functions for `prior collections`, `posterior collections`, `Laplace posteriors`, `baseline corrections`, and `MCMC results`.

Input:
	* `smd` is the input SMD object from where the data will be read
	* `i` is the number of the molecule to from which to read the data

"""

import numpy as _np
from ._general_smd import smd_io_error

def priors(smd,i):
	"""
	Load a BIASD prior distribution collection from an SMD.
	
	Returns:
		* a `biasd.distributions.parameter_collection`
	"""
	
	attr = smd.data[i].attr.__dict__
	key1 = 'biasd_priors_names'
	key2 = 'biasd_priors_parameters'
	if attr.has_key(key1) and attr.has_key(key2):
		from ..distributions import parameter_collection
		return parameter_collection.new_from_smd(attr.get(key1),attr.get(key2))
	else:
		raise smd_io_error()
		return None
	
def posterior(smd,i):
	"""
	Load a BIASD posterior distribution collection from an SMD.
	
	Returns:
		* a `biasd.distributions.parameter_collection`
	"""
	
	attr = smd.data[i].attr.__dict__
	key1 = 'biasd_posterior_names'
	key2 = 'biasd_posterior_parameters'
	if attr.has_key(key1) and attr.has_key(key2):
		from ..distributions import parameter_collection
		return parameter_collection.new_from_smd(attr.get(key1),attr.get(key2))
	else:
		raise smd_io_error()
		return None

def laplace_posterior(smd,i):
	"""
	Load a Laplace posterior object from an SMD.
	
	Returns:
		* a `biasd.laplace._laplace_posterior`
	"""
	attr = smd.data[i].attr.__dict__
	key1 = 'biasd_laplace_posterior_mu'
	key2 = 'biasd_laplace_posterior_covar'
	if attr.has_key(key1) and attr.has_key(key2):
		from ..laplace import _laplace_posterior
		return _laplace_posterior(_np.array(attr.get(key1)),_np.array(attr.get(key2)))
	else:
		raise smd_io_error()
		return None

def baseline(smd,i):
	"""
	Load a baseline calculation from an SMD.
	
	Returns:
		* a `biasd.utils.baseline.params`
	"""
	attr = smd.data[i].attr.__dict__
	keys= ['baseline_' + j for j in ['pi','mu','var','r','baseline','r2','log_likelihood','iterations']]

	if _np.all([attr.has_key(keyi) for keyi in keys]):
		from ..utils.baseline import params
		return params(*[_np.array(attr.get(keyi)) for keyi in keys])
	else:
		raise smd_io_error()
		return None
		
def mcmc(smd,i):
	"""
	Load a BIASD MCMC result from an SMD.
	
	Returns:
		* a `biasd.smd.read.mcmc.mcmc_result`
	"""
	
	class mcmc_result(object):
		def __init__(self, acor,chain,lnprobability,iterations,naccepted,samples):
			self.acor = acor
			self.chain = chain
			self.lnprobability = lnprobability
			self.iterations = iterations
			self.naccepted = naccepted
			self.samples = samples
			
	attr = smd.data[i].attr.__dict__
	keys= ['mcmc_' + j for j in ['acor','chain','lnprobability','iterations','naccepted','samples']]
	if _np.all([attr.has_key(keyi) for keyi in keys]):
		return mcmc_result(*[_np.array(attr.get(keyi)) for keyi in keys])
	else:
		raise smd_io_error()
		return None