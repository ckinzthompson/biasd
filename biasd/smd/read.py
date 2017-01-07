"""
Reads a BIASD result from an HDF5 SMD file. This includes functions for `parameter collections`, `Laplace posteriors`, `baseline corrections`, and `MCMC results`.

Input:
	* `group` is the input HDF5 group from where the data will be read
Returns:
	* The data formatted as an object
"""

def _safely(fxn):
	from .smd_hdf5 import smd_malformed
	def wrapper(*args,**kwargs):
		try:
			return fxn(*args,**kwargs)
		except:
			raise smd_malformed
	return wrapper

@_safely
def parameter_collection(group,label='BIASD parameter collection'):
	"""
	Load a BIASD prior distribution collection from an HDF5 group.
	
	Returns:
		* a `biasd.distributions.parameter_collection`
	"""
	assert group.attrs['description'] == label
	from ..distributions import parameter_collection,beta,dirichlet,empty,gamma,normal,uniform

	l = ['e1','e2','sigma','k1','k2']
	dist_dict = {'beta':beta, 'dirichlet':dirichlet, 'empty':empty, 'gamma':gamma, 'normal':normal, 'uniform':uniform}
	
	return parameter_collection(*[dist_dict[group.attrs[ll+' distribution type']](*group.attrs[ll + ' parameters']) for ll in l])

@_safely
def laplace_posterior(group,label = 'BIASD Laplace posterior'):
	"""
	Load a Laplace posterior object from an HDF5 group.
	
	Returns:
		* a `biasd.laplace._laplace_posterior`
	"""
	
	assert group.attrs['description'] == label
	from ..laplace import _laplace_posterior
	keys = ['mu','covar']
	return _laplace_posterior(*[group.attrs[i] for i in keys])

	
@_safely
def baseline(group,label = 'White-noise baseline correction parameters'):
	"""
	Load a baseline calculation from an HDF5 group.
	
	Returns:
		* a `biasd.utils.baseline._params`
	"""
	assert group.attrs['description'] == label
	from ..utils.baseline import _params
	keys = ['pi','mu','var','R2','log likelihood','iterations']
	p = [group.attrs[i] for i in keys]
	p.insert(3,group['r'].value)
	p.insert(4,group['baseline'].value)
	return _params(*p)
		
@_safely
def mcmc(group,label='BIASD MCMC result'):
	"""
	Load a BIASD MCMC result from an HDF5 group.
	
	Returns:
		* a `biasd.mcmc.mcmc_result`
	"""
	assert group.attrs['description'] == label
	from ..mcmc import mcmc_result
	keys = ['autocorrelation times', 'iterations', 'number accepted', 'number of walkers','number of data dimensions']
	p = [group.attrs[i] for i in keys]
	p.insert(1,group['sampler chain'].value)
	p.insert(2,group['log probability'].value)
	return mcmc_result(p)

@_safely
def kmeans(group,label="K-means result"):
	assert group.attrs['description'] == label
	from ..distributions.kmeans import _results_kmeans
	keys = ['nstates','pi','mu','var']
	p = [group.attrs[i] for i in keys]
	p.insert(2,group['responsibilities'].value)
	return _results_kmeans(*p)
	