"""
.. module:: laplace
	:synopsis: Contains function to calculate the laplace approximation to the BIASD posterior probability distribution.

"""

import numpy as np
from scipy.optimize import minimize
from .likelihood import log_posterior
from .distributions import parameter_collection,normal,convert_distribution

deftol = 10*np.sqrt(np.finfo(np.float64).eps)

def calc_hessian(fxn,x,eps=deftol):
	"""
	Calculate the Hessian using the finite difference approximation.

	Finite difference formulas given in Abramowitz & Stegun

		- Eqn. 25.3.24 (on-diagonal)
		- Eqn. 25.3.27 (off-diagonal)

	Input:
		* `fxn` is a function that can be evaluated at x
		* `x` is a 1D `np.ndarray`

	Returns:
		* an NxN `np.ndarray`, where N is the size of `x`

	"""

	# Notes:
	# xij is the position to evaluate the function at
	# yij is the function evaluated at xij
	#### if i or j = 0, it's the starting postion
	#### 1 or m1 are x + 1.*eps and x - 1.*eps, respetively


	h = np.zeros((x.size,x.size))
	y00 = fxn(x)

	for i in range(x.size):
		for j in range(x.size):
			#Off-diagonals below the diagonal are the same as those above.
			if j < i:
				h[i,j] = h[j,i]
			else:
				#On-diagonals
				if i == j:
					x10 = x.copy()
					xm10 = x.copy()
					x20 = x.copy()
					xm20 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x20[i] += 2*eps
					xm20[i] -= 2*eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y20 = fxn(x20)
					ym20 = fxn(xm20)

					h[i,j] = eps**(-2.)/12. * (-y20 + 16.* y10 - 30.*y00 +16.*ym10 - ym20)

				#Off-diagonals above the diagonal
				elif j > i:
					x10 = x.copy()
					xm10 = x.copy()
					x01 = x.copy()
					x0m1 = x.copy()
					x11 = x.copy()
					xm1m1 = x.copy()

					x10[i] += eps
					xm10[i] -= eps
					x01[j] += eps
					x0m1[j] -= eps
					x11[i] += eps
					x11[j] += eps
					xm1m1[i] -= eps
					xm1m1[j] -= eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)
					y01 = fxn(x01)
					y0m1 = fxn(x0m1)
					y11 = fxn(x11)
					ym1m1 = fxn(xm1m1)

					h[i,j] = -1./(2.*eps**2.) * (y10 + ym10 + y01 + y0m1 - 2.*y00 - y11 - ym1m1)
	return h

class _laplace_posterior:
	"""
	Holds the results of a Laplace approximation of the posterior probability distribution from BIASD
	"""
	def __init__(self,mean,covar,prior=None):
		self.mu = mean
		self.covar = covar
		self.posterior = parameter_collection(*[normal(m,s) for m,s in zip(self.mu,self.covar.diagonal()**.5)])

	def transform(self,prior):
		self.posterior.e1 = convert_distribution(self.posterior.e1,prior.e1.name)
		self.posterior.e2 = convert_distribution(self.posterior.e2,prior.e2.name)
		self.posterior.sigma = convert_distribution(self.posterior.sigma,prior.sigma.name)
		self.posterior.k1 = convert_distribution(self.posterior.k1,prior.k1.name)
		self.posterior.k2 = convert_distribution(self.posterior.k2,prior.k2.name)

	def samples(self,n):
		return np.random.multivariate_normal(self.mu,self.covar,n)

def _min_fxn(theta,data,prior,tau,device):
	try:
		return -1.*log_posterior(theta,data,prior,tau,device)
	except:
		print(theta)
		raise Exception('There is an error in the likelihood calculation -- 99 per cent chance it is a numba issue')


def _minimizer(inputt):
	data,prior,tau,x0,meth,device = inputt
	mind =  minimize(_min_fxn,x0,method=meth,args=(data,prior,tau,device), tol=deftol, options={'maxiter':1000})
	return mind

def find_map(data,prior,tau,meth='nelder-mead',guess=None,nrestarts=1,device=0):
	'''
	Use numerical minimization to find the maximum a posteriori estimate of a BIASD log-posterior distribution.

	Inputs:
		* `data` is a 1D `np.ndarray` of the time series
		* `prior` is a `biasd.distributions.parameter_collection` that contains the prior the BIASD Bayesian inference
		* `tau` is the measurement period

	Optional:
		* `meth` is the minimizer used to find the minimum of the negative posterior (i.e., the maximum). Defaults to simplex.
		* `xx` will initialize the minimizer at this theta position. Defaults to mean of the priors.

	Returns:
		* the minimizer dictionary
	'''

	#If no xx, start at the mean of the priors
	if guess is None:
		guess = prior.rvs(1).flatten()
		for _ in range(1000):
			guess = prior.rvs(1).flatten()
			_min_fxn(guess,data,prior,tau,device)

	for iteration in range(nrestarts):
		out = _minimizer([data,prior,tau,guess,meth,device,])
		guess = out.x
		# print(f"iteration {iteration}:",xx)

	return out.success,out.x

def laplace_approximation(data,prior,tau,guess = None, nrestarts=1,verbose=False,ensure=False,device=0,epsilon=deftol):
	'''
	Perform the Laplace approximation on the BIASD posterior probability distribution of this trace.

	Inputs:
		* `data` is a 1D `np.ndarray` of the time series
		* `prior` is a `biasd.distributions.parameter_collection` that contains the prior the BIASD Bayesian inference
		* `tau` is the measurement period

	Optional:
		* `nrestarts` is the number of times to try to find the MAP in `find_map`.

	Returns:
		* a `biasd.laplace._laplace_posterior` object, which has a `.mu` with the means, a `.covar` with the covariances, and a `.posterior` which is a marginalized `biasd.distributions.parameter_collection` of normal distributions.
	'''

	if verbose:	print('Laplace Approximation')

	#Calculate the best MAP estimate
	import time
	t0 = time.time()
	success, mapx = find_map(data,prior,tau,guess=guess,device=device)
	t1 = time.time()
	if verbose:
		print(f"MAP: {t1-t0} s")
	print(mapx,success)

	#Calculate the Hessian at MAP estimate
	if success:
		t0 = time.time()
		hessian = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mapx, eps=epsilon)
		t1 = time.time()
		if verbose:
			print(f"Hessian: {t1-t0} s")

		if ensure:
			#Ensure that the hessian is positive semi-definite by checking that all eigenvalues are positive
			#If not, expand the value of machine error in the hessian calculation and try again
			try:
				#Check eigenvalues, use pseudoinverse if ill-conditioned
				var = -np.linalg.inv(hessian)

				#Ensuring Hessian(variance) is stable
				epsilon *= 2
				new_hess = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mapx,eps= epsilon)
				new_var = -np.linalg.inv(new_hess)
				
				iter = 0
				while np.any(np.abs(new_var-var)/var > 1e-2):
					epsilon *= 2
					var = new_var.copy()
					new_hess = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mapx,eps= epsilon)
					new_var = -np.linalg.inv(new_hess)
					iter +=1
					# 2^26 times feps = 1. Arbitrary upper-limit, increase if necessary (probably not for BIASD)
					if iter > 25:
						raise ValueError('Whelp, you hit the end there. bud')
				if verbose:
					print(f'Hessian iterations {iter}: {epsilon}')

				#Ensure symmetry of covariance matrix if witin machine error
				if np.allclose(var,var.T):
					n = var.shape[0]
					var = np.tri(n,n,-1)*var+(np.tri(n,n)*var).T
					return _laplace_posterior(mapx,var)

			#If this didn't work, return None
			except:
				raise ValueError("Wasn't able to calculate the Hessian")
			# raise Exception('Laplace Failure: Not symmetric')
		
		var = -np.linalg.inv(hessian)
		n = var.shape[0]
		var = np.tri(n,n,-1)*var+(np.tri(n,n)*var).T
		if not np.allclose(var,var.T):
			print('Inverse Hessian is not (numerically) positive semi-definite')
		return _laplace_posterior(mapx,var)
	

# def predictive_from_samples(x,samples,tau,device=0):
# 	'''
# 	Returns the posterior predictive distribution calculated from samples -- the average value of the likelihood function evaluated at `x` marginalized from the samples of BIASD parameters given in `samples`.

# 	Samples can be generated from a posterior probability distribution. For instance, after a Laplace approximation, just draw random variates from the multivariate-normal distribution -- i.e., given results in `r`, try `samples = np.random.multivariate_normal(r.mu,r.covar,100)`. Alternatively, some posteriors might already have samples (e.g., from MCMC).

# 	Input:
# 		* `x` a `np.ndarry` where to evaluate the likelihood at (e.g., [-.2 ... 1.2] for FRET)
# 		* `samples` is a (N,5) `np.ndarray` containing `N` samples of BIASD parameters (e.g. \Theta)
# 		* `tau` the time period with which to evaluate the likelihood function
# 	Returns:
# 		* `y` a `np.ndarray` the same size as `x` containing the marginalized likelihood function evaluated at x
# 	'''
# 	n = samples.shape[0]
# 	y = np.mean([np.exp(nosum_log_likelihood(samples[i],x,tau,device=device)) for i in range(n)],axis=0)
# 	return y

