"""
Add a BIASD result to an SMD file. This includes functions for `prior collections`, `posterior collections`, `Laplace posteriors`, `baseline corrections`, and `MCMC results`.

Input:
	* `smd` is the input SMD object where the data will be added
	* `i` is the number of the molecule to which to add the data
	* `final argument` is the data that will be added

Returns:
	* `smd` is an updated SMD object
"""

from ._general_smd import smd_io_error

def priors(smd,i,priors):
	"""
	Piror distribution collection from biasd.distributions
	"""
	smd.data[i].attr.biasd_priors_names, smd.data[i].attr.biasd_priors_parameters = priors.format_for_smd()
	return smd

def posterior(smd,i,posterior):
	"""
	Posterior distribution collection from biasd.distributions
	"""
	smd.data[i].attr.biasd_posterior_names, smd.data[i].attr.biasd_posterior_parameters = posterior.format_for_smd()
	return smd

def laplace_posterior(smd,i,lp):
	"""
	Laplace posterior from biasd.laplace
	"""
	smd.data[i].attr.biasd_laplace_posterior_mu = lp.mu.tolist()
	smd.data[i].attr.biasd_laplace_posterior_covar = lp.covar.tolist()
	return smd

def baseline(smd,i,p):
	"""
	Baseline result from biasd.utils.baseline
	"""
	# pi,mu,var,r,baseline,R2,ll,iter
	smd.data[i].attr.baseline_pi = p.pi.tolist()
	smd.data[i].attr.baseline_mu = p.mu.tolist()
	smd.data[i].attr.baseline_var = p.var.tolist()
	smd.data[i].attr.baseline_r = p.r.tolist()
	smd.data[i].attr.baseline_baseline = p.baseline.tolist()
	smd.data[i].attr.baseline_r2 = p.r2.tolist()
	smd.data[i].attr.baseline_log_likelihood = p.log_likelihood
	smd.data[i].attr.baseline_iterations = p.iterations
	return smd

def mcmc(smd,i,sampler):
	"""
	MCMC Result from biasd.mcmc
	"""
	# acor (maybe),chain,lnprobability,iterations,naccepted
	if not sampler.__dict__.has_key('acor'):
		sampler.get_autocorr_time()
	smd.data[i].attr.mcmc_acor = sampler.acor.tolist()
	smd.data[i].attr.mcmc_chain = sampler.chain.tolist()
	smd.data[i].attr.mcmc_lnprobability = sampler.lnprobability.tolist()
	smd.data[i].attr.mcmc_iterations = sampler.iterations
	smd.data[i].attr.mcmc_naccepted = sampler.naccepted.tolist()
	from ..mcmc import get_samples
	smd.data[i].attr.mcmc_samples = get_samples(sampler).tolist()
	return smd

