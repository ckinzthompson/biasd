"""
.. module:: baseline

	:synopsis: Contains functions to correct for white-noise baseline drift

"""

from scipy.linalg import solve_banded
import numpy as np
from .clustering import GMM_EM_1D,kmeans

def banded_solver(a,b,c,d):
	"""
	Fast tri-diagonal matrix algebra solver. Solves Ax = B where A is a tri-diagonal matrix.
	
	Input:
		* `a` is the lower-diagonal of `A`
		* `b` is the diagonal of `A`
		* `c` is the upper-diagonal of `A`
		* `d` is the RHS (`B`)
	
	Returns:
		* `x`, which is the solution
	"""
	banded = np.vstack((c,b,a))
	return solve_banded((1,1),banded,d)
	
def simulate_diffusion(n,sigma):
	"""
	Returns a n-long randomwalk with normal distribution steps of std dev sigma
	"""
	d = np.random.normal(size=n)*sigma
	return d.cumsum()

def solve_baseline(R2,d):
	"""
	Expectation Step - calculate best baseline
	"""
	tdm_a = np.repeat(-1.,d.size)
	tdm_b = np.repeat(R2 + 2,d.size)
	tdm_c = np.repeat(-1.,d.size)
	tdm_b[0] = 1. + R2
	tdm_b[-1] = 1. + R2
	tdm_d = R2 * (d)
	# bstar = TDMASolve(tdm_a,tdm_b,tdm_c, tdm_d)
	bstar = banded_solver(tdm_a,tdm_b,tdm_c, tdm_d)
	return bstar #- bstar[0]

def optimal_r2(r2,baseline,sk2):
	"""
	Maximization Step - returns R2
	"""
	from scipy.optimize import root
	def minr2fxn(r2,baseline,t,sk2):
		if r2 > 1e-1 or r2 < 1e-15:
			return np.inf
		b = ((np.roll(baseline,-1)[:-1]-baseline[:-1])**2.).sum()
		f = b/(2.*sk2*r2**2.)
		num = 1.+ (2.+r2) / np.sqrt(4.*r2 + r2**2.)
		denom = 2. + r2 + np.sqrt(4.*r2 + r2**2.)  + 1e-300
		f -= t* (num/denom)
		return f
	r2 = root(minr2fxn,x0 = np.array(r2), args=(baseline,float(baseline.size),sk2),options={'maxfev':10000}).x
	# return r2
	return r2[0]

def gaussian(x,mu,var):
	return 1./np.sqrt(2.*np.pi*var) * np.exp(-.5/var*(x - mu)**2.)
	
def estep(x,pi,mu,var):
	"""
	Expectation Step - returns responsibilities
	"""
	r = pi[None,:] * gaussian(x[:,None], mu[None,:],var[None,:])
	# r = pi[None,:] * gaussian(x[:,None], mu[None,:],var)
	r = r/(1e-300+r.sum(1)[:,None])
	return r
	
def mstep(x,r):
	"""
	Maximization Step - returns fractions, means, and variances
	"""
	n = np.sum(r,axis=0) + 1e-300
	mu = 1./n * np.sum(r*x[:,None],axis=0)
	var= 1./n * np.sum(r * (x[:,None] - mu[None,:])**2.,axis=0)
	# var= np.sum(1./n * np.sum(r * (x[:,None] - mu[None,:])**2.,axis=0))
	pi = n / n.sum()
	# var = (pi*var).sum()
	return pi,mu,var
	
def ideal_signal(r,mu):
	"""
	Generate the weighted estimate of the signal
	"""
	return (r*mu[None,:]).sum(1)

class params(object):
	"""
	Container for remove_baseline results
	"""
	def __init__(self,pi,mu,var,r,baseline,R2,ll,iters):
		self.pi = pi
		self.mu = mu
		self.var = var
		self.baseline = baseline
		self.r2 = R2
		self.log_likelihood = ll
		self.iterations = iters
		self.responsibilities = r
		self.r = r
	

def remove_baseline(d,R2=None,nstates=2,maxiter=1000,relative_threshold = 1e-20):
	"""
	Removes the baseline sort of according to:

	:Title: 
		Automated Maximum Likelihood Separation of Signal from Baseline in Noisy Quantal Data
	:Authors:
		Bruno, WJ
		Ullah, G
		Mak, DD
		Pearson JE
	:Citation:
		Biophys. J. 2013, 105, 68-79.
	
	Input:
		* `d` is a 1D array of the signal
		* `R2` can be an initial guess for the ratio of the random walk's variance to that of the noise variance of the underlying states.
		* `nstates` should probably be two for BIASD
		* `maxiter` the maximum number of times 
		* `relative_threshold` is the convergence threshold for the releative change in the log-likelihood function. Smaller values are more rigorous.
	Returns:
		* `baseline_results` which is a `biasd.utils.baseline.params` object. You can access the baseline with `baseline_results.basline`
	"""
	
	# Use the provided guess for R2
	if isinstance(R2,float):
		bstar = solve_baseline(R2,d)
	# Or... estimate it using filters
	else:
		from scipy.ndimage import gaussian_filter,minimum_filter1d
		bstar = gaussian_filter(minimum_filter1d(d,25),50)
		# just so R2 is initialized
		R2 = 1e-12
	
	### k-means initialization
	# return params(0,0,0,0,bstar,0,0,0)
	# ik = kmeans(d-bstar,nstates)
	# r = ik.r
	# mu = ik.mu[...,0]
	# var = ik.var[...,0,0]
	# pi = ik.pi

	### GMM initialization
	ig = GMM_EM_1D(d-bstar,nstates,init_kmeans=True)
	r = ig.r
	mu = ig.mu
	var = ig.var
	pi = ig.pi
	
	# Zero...
	bstar += mu.min()
	mu -= mu.min()
	
	# Clean up initialization
	r = estep(d - bstar,pi,mu,var)
	pi,mu,var = mstep(d-bstar,r)
	E_var = ((r*var[None,:]).sum(0)/(np.sum(r,axis=0)+1e-300) * pi).sum()
	R2 = optimal_r2(R2,bstar,E_var)
	
	# Initialize log-likelihoods
	l0 = -np.inf
	ll = -np.inf

	# Start the EM iterations
	for iteration in range(maxiter):
		# Solve baseline
		bstar = solve_baseline(R2,d-ideal_signal(r,mu))
		
		# Compute responsibilities
		r = estep(d - bstar,pi,mu,var)
		
		# Maximize parameters
		pi,mu,var = mstep(d-bstar,r)
		
		#Calcualte expectation of variance
		E_var = ((r*var[None,:]).sum(0)/(np.sum(r,axis=0)+1e-300) * pi).sum()

		# zero lowest state.
		bstar += mu.min()
		mu -= mu.min()

		# Maximize R2
		R2 = optimal_r2(R2,bstar,E_var)
		vb = R2*E_var
		
		# Update log-likelihood
		l0 = ll
		ll = (np.sum(np.log(np.sum(pi[None,:] * gaussian(d[:,None] - bstar[:,None],mu[None,:],var[None,:]),axis=1))) + np.sum(-.5*np.log(2.*np.pi*vb) -.5/vb*(np.roll(bstar,-1)[:-1]-bstar[:-1])**2.))
		
		# Check for convergence
		llratio = np.abs((ll - l0)/l0)
		if llratio < relative_threshold:
			break

	# Return results	
	results = params(pi,mu,var,r,bstar,R2,ll,iteration)
	if iteration > maxiter - 1:
		raise DataWarning("Didn't converge in time")
	return results