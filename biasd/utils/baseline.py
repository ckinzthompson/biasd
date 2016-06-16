"""
Baseline remover 

Use this to remove the baseline from a trace that it can be better processed using BIASD.

import matplotlib.pyplot as plt
from ckt_utils import sm_sim

dd = sm_sim.simtrace(np.array(((0.,3.),(8.,0.))),np.array((.0,1)),.05,1000,.01)
signal = dd.y
baseline = simulate_diffusion(signal.size,.005)
d = signal + baseline + np.random.normal(size=signal.size)*.02

p = remove_baseline(d)
f,a = plt.subplots(2,sharex=True)
a[0].plot(d)
a[0].plot(p.baseline,'r')
a[1].plot(d-p.baseline)
plt.show()
"""


import numpy as np

def TDMASolve(a, b, c, d):
	"""
	Tri-diagonal Matrix Algebra solver.
	Solves Ax = B where A is a tri-diagonal matrix with
	a - lower diagonal
	b - diagonal
	c - upper diagonal
	
	d - RHS
	
	Returns x
	"""
	# I rewrote the C version from... https://en.wikibooks.org/wiki/Algorithm_Implementation/Linear_Algebra/Tridiagonal_matrix_algorithm
	x = np.copy(d)

	c[0] /= b[0]
	x[0] /= b[0]

	for i in range(1,d.size):
		m = (b[i] - a[i]*c[i-1])
		c[i] /= m
		x[i] = (x[i] - a[i]*x[i-1])/m

	for i in range(d.size-1)[::-1]:
		x[i] -= c[i]*x[i+1]
	return x
	
	
def simulate_diffusion(n,sigma):
	"""
	Returns a n-long randomwalk with normal distribution steps of std dev sigma
	"""
	d = np.random.normal(size=n)*sigma
	return d.cumsum()

def kmeans(x,nstates,nrestarts=1):
	"""
	K-means Clustering
	x is Nxd
	----
	Returns pi_k, r_nk, mu_k, sig_k
	"""

	if x.ndim == 1:
		x = x[:,None]

	jbest = np.inf
	mbest = None
	rbest = None
	for nr in range(nrestarts):
		mu_k = x[np.random.randint(0,x.shape[0],size=nstates)]
		j_last = np.inf
		for i in range(500):
			dist = np.sqrt(np.sum(np.square(x[:,None,:] - mu_k[None,...]),axis=2))
			r_nk = (dist == dist.min(1)[:,None]).astype('i')
			j = (r_nk.astype('f') * dist).sum()
			mu_k = (r_nk[:,:,None].astype('f')*x[:,None,:]).sum(0)/(r_nk.astype('f').sum(0)[:,None]+1e-16)
			if np.abs(j - j_last)/j <= 1e-100:
				if j < jbest:
					jbest = j
					mbest = mu_k
					rbest = r_nk
				break
			else:
				j_last = j
	mu_k = mbest
	r_nk = rbest
	sig_k = np.empty((nstates,x.shape[1],x.shape[1]))
	for k in range(nstates):
		sig_k[k] = np.cov(x[r_nk[:,k]==1.].T)
	pi_k = (r_nk.sum(0)).astype('f')
	pi_k /= pi_k.sum()

	xsort = pi_k.argsort()[::-1]
	#pi_k is fraction, r_nk is responsibilities, mu_k is means, sig_k is variances
	return [pi_k[xsort],r_nk[:,xsort],mu_k[xsort],sig_k[xsort]]

def solve_baseline(R2,d):
	tdm_a = np.repeat(-1.,d.size)
	tdm_b = np.repeat(R2 + 2,d.size)
	tdm_c = np.repeat(-1.,d.size)
	tdm_b[0] = 1. + R2
	tdm_b[-1] = 1. + R2
	tdm_d = R2 * (d)
	bstar = TDMASolve(tdm_a,tdm_b,tdm_c, tdm_d)
	return bstar #- bstar[0]

def optimal_r2(r2,baseline,sk2):
	from scipy.optimize import root
	def minr2fxn(r2,baseline,t,sk2):
		if r2 > 1e2 or r2 < 1e-8:
			return np.inf
		b = ((np.roll(baseline,-1)[:-1]-baseline[:-1])**2.).sum()
		f = b/(2.*sk2*r2**2.)
		num = 1.+ (2.+r2) / np.sqrt(4.*r2 + r2**2.)
		denom = 2. + r2 + np.sqrt(4.*r2 + r2**2.) 
		f -= t* (num/denom)
		return f
	r2 = root(minr2fxn,x0 = np.array(r2), args=(baseline,float(baseline.size),sk2),options={'maxfev':10000}).x
	return r2

def gaussian(x,mu,var):
	return 1./np.sqrt(2.*np.pi*var) * np.exp(-.5/var*(x - mu)**2.)
	
def estep(x,pi,mu,var):
	"""
	Expectation Step - returns responsibilities
	"""
	r = pi[None,:] * gaussian(x[:,None], mu[None,:],var[None,:])
	r = r/(1e-300+r.sum(1)[:,None])
	return r
	
def mstep(x,r):
	"""
	Maximization Step - returns fractions, means, and variances
	"""
	n = np.sum(r,axis=0) + 1e-300
	mu = 1./n * np.sum(r*x[:,None],axis=0)
	var= 1./n * np.sum(r * (x[:,None] - mu[None,:])**2.,axis=0)
	pi = n / n.sum()
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


def remove_baseline(d,R2=0.01,maxiter=1000,relative_threshold = 1e-5):
	"""
	Removes the baseline sort of according to:
	--------------
	Automated Maximum Likelihood Separation of Signal from Baseline in
	Noisy Quantal Data
	
	Bruno, WJ, Ullah, G, Mak, DD, Pearson JE
	Biophys J 2013, 105, 68-79.
	--------------
	
	d is a 1d array of the signal
	R2 is an initial guess for the ratio of the random walk's variance to that of the noise variance of the underlying states.
	"""
	
	R2 = .01
	bstar = solve_baseline(R2,d)

	pi,_,mu,sig = kmeans(d-bstar,2)
	mu = mu[:,0]
	var = sig[:,0,0]**2.

	l0 = -np.inf
	ll = -np.inf
	r = estep(d - bstar,pi,mu,var)

	for iteration in range(maxiter):
		bstar = solve_baseline(R2,d-ideal_signal(r,mu))
		r = estep(d - bstar,pi,mu,var)
		pi,mu,var = mstep(d-np.nan_to_num(bstar),r)

		bstar += mu.min()
		mu -= mu.min()
		# worst case scenario
		R2 = optimal_r2(R2,bstar,var.max())
		if R2 > 1: # HACK b/c sometimes you hemorrhage R2... 
			R2 /= 10**(np.random.rand()*10)
		vb = R2*(var*pi).sum()
		l0 = ll
		ll = (np.sum(np.log(np.sum(pi[None,:] * gaussian(d[:,None] - bstar[:,None],mu[None,:],var[None,:]),axis=1))) + np.sum(-.5*np.log(2.*np.pi*vb) -.5/vb*(np.roll(bstar,-1)[:-1]-bstar[:-1])**2.))[0]
		llratio = np.abs((ll - l0)/l0)
		if llratio < relative_threshold:
			break	
	return params(pi,mu,var,r,bstar,R2,ll,iteration)
