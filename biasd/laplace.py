import numpy as np
from scipy.optimize import minimize
from biasd.likelihood import log_posterior
from biasd.distributions import parameter_collection,normal,convert_distribution

def calc_hessian(fxn,x,eps = np.sqrt(np.finfo(np.float64).eps)):
	"""
	Finite difference approximation of the Hessian
	Using Abramowitz & Stegun Eqn. 25.3.23 (on-diagonal), and 25.3.26 (off-diagonal)

	-- xij is the position to evaluate the function at
	-- if i or j = 0, it's the starting postion, 1 or m1 are x + 1.*eps and x - 1.*eps, respetively
	-- yij is the function evaluated at xij
	"""
	
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

					x10[i] += eps
					xm10[i] -= eps

					y10 = fxn(x10)
					ym10 = fxn(xm10)

					h[i,j] = eps**(-2.) * (y10 - 2.*y00 + ym10)

				#Off-diagonals above the diagonal
				elif j > i:
					x11 = x.copy()
					x1m1 = x.copy()
					xm1m1 = x.copy()
					xm11 = x.copy()

					x11[i] += eps
					x11[j] += eps
					x1m1[i] += eps
					x1m1[j] -= eps
					xm1m1[i] -= eps
					xm1m1[j] -= eps
					xm11[i] -= eps
					xm11[j] += eps

					y11 = fxn(x11)
					y1m1 = fxn(x1m1)
					ym1m1 = fxn(xm1m1)
					ym11 = fxn(xm11)

					h[i,j] = 1./(4.*eps**2.) * (y11 - y1m1 - ym11 + ym1m1)
	return h
	
class _laplace_posterior:
	"""
	Holds the results of a laplace approximation of the posterior probability distribution from BIASD
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
		
		
def find_map(data,prior,tau,meth='nelder-mead',xx=None,nrestarts=2):
	'''
	Use numerical minimization to find the maximum a posteriori estimate of the posterior
	
	
	Provide xx to force first theta initialization at that theta
	meth is the method used by the minimizer - default to simplex
	nrestarts is the number of restarts for the MAP
	'''

	ylist = []

	#If no xx, start at the mean of the priors
	if not isinstance(xx,np.ndarray):
		xx = prior.mean()
	
	#Rounds to minimze the -log posterior
	ylist.append(minimize(lambda theta: -1.*log_posterior(theta,data,prior,tau),x0=xx,method=meth))
	
	for i in range(nrestarts):
		#Try a random location consistent with the prior.
		xx = prior.rvs(1)
		ylist.append(minimize(lambda theta: -1.*log_posterior(theta,data,prior,tau),x0=xx,method=meth))
	
	#Select the best MAP estimate
	ymin = np.inf
	for i in ylist:
		if i['success']:
			if i['fun'] < ymin:
				ymin = i['fun']
				y = i
	#If no MAP estimates, return None
	if ymin == np.inf:
		y = None
	return y

def laplace_approximation(data,prior,tau,nrestarts=2,verbose=False):
	'''
	Perform the Laplace approximation on the posterior probability distribution of this trace
	'''

	#Calculate the best MAP estimate
	import time
	t0 = time.time()
	mind = find_map(data,prior,tau,nrestarts=nrestarts)
	t1 = time.time()
	if verbose:
		print t1-t0

	if not mind is None:
		#Calculate the Hessian at MAP estimate
		if mind['success']: 
			mu = mind['x']
			feps = np.sqrt(np.finfo(np.float).eps)
			t0 = time.time()
			hessian = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mu,eps=feps)
			t1 = time.time()
			if verbose:
				print t1-t0
			#Ensure that the hessian is positive semi-definite by checking that all eigenvalues are positive
			#If not, expand the value of machine error in the hessian calculation and try again
			try:
				#Check eigenvalues
				while np.any(np.linalg.eig(-hessian)[0] <= 0.):
					feps *= 2.
					#Calculate hessian
					t0 = time.time()
					hessian = calc_hessian(lambda theta: log_posterior(theta,data,prior,tau), mu,eps=feps)
					t1 = time.time()
					if verbose:
						print t1-t0
				#Invert hessian to get the covariance matrix
				var = np.linalg.inv(-hessian)
				#Ensure symmetry of covariance matrix if witin machine error
				if np.allclose(var,var.T):
					var = np.tri(5,5,-1)*var+(np.tri(5,5)*var).T
					return _laplace_posterior(mu,var)
				
			#If this didn't work, return None
			except np.linalg.LinAlgError:
				raise ValueError("Wasn't able to calculate the Hessian")
				pass
	raise ValueError("No MAP estimate")
	return _laplace_posterior(None,None)
