import numpy as np

class _results_kmeans(object):
	def __init__(self,nstates,pi,r,mu,var):
		self.nstates = nstates
		self.pi = pi
		self.r = r
		self.mu = mu
		self.var = var
	
	def sort(self):
		xsort = self.pi.argsort()[::-1]
		# xsort = np.argsort(self.mu)
		self.pi = self.pi[xsort]
		self.r = self.r[:,xsort]
		self.var = self.var[xsort]
		self.mu = self.mu[xsort]
		
class _results_gmm(_results_kmeans):
	def __init__(self,nstates,pi,r,mu,var,ll):
		self.nstates = nstates
		self.pi = pi
		self.r = r
		self.mu = mu
		self.var = var
		self.ll = -np.inf
		self.ll_last = -np.inf

def kmeans(x,nstates,nrestarts=1):
	"""
	Multidimensional, K-means Clustering
	
	Input:
		* `x` is an (N,d) `np.array`, where N is the number of data points and d is the dimensionality of the data
		* `nstates` is the K in K-means
		* `nrestarts` is the number of times to restart. The minimal variance results are provided

	Returns:
		* a `biasd.utils.clustering._results_kmeans` object that contains
			- `pi_k`, the probability of each state
			- `r_nk`, the responsibilities of each data point
			- `mu_k` the means
			- `var_k` the covariances
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

	#pi_k is fraction, r_nk is responsibilities, mu_k is means, sig_k is variances
	results = _results_kmeans(nstates,pi_k,r_nk,mu_k,sig_k**2.)
	results.sort()
	return results


def GMM_EM_1D(x,k=2,maxiter=1000,relative_threshold=1e-6,init_kmeans=True):
	"""
	One-dimensional, Gaussian Mixture Model Clustering with expectation-maximization algorithm.
	
	Input:
		* `x` is an (N,d) `np.array`, where N is the number of data points and d is the dimensionality of the data
		* `nstates` is the number of states
		* `maxiter` is the maximum number of iterations
		* `relative_threshold` is the convergence criteria for the relative change in the log-likelihood
		* `init_kmeans` is a boolean for whether to initialize the GMM with a K-means pass
	
	Returns:
		* a `biasd.utils.clustering._results_gmm` object that contains
			- `pi_k`, the probability of each state
			- `r_nk`, the responsibilities of each data point
			- `mu_k` the means
			- `var_k` the covariances
	"""


	# Make sure x is proper
	if not isinstance(x,np.ndarray) or x.ndim != 1:
		raise ValueError('Input is not really a 1D ndarray, is it?')
		return None
	
	def Nk_gaussian(x,mu,var):
		return 1./np.sqrt(2.*np.pi*var[None,:]) * np.exp(-.5/var[None,:]*(x[:,None] - mu[None,:])**2.)
	
	# Initialize
	if init_kmeans:
		ik = kmeans(x,k,nrestarts=5)
		mu_k = ik.mu[:,0] # slicing b/c it's 1D not ND here...
		var_k = ik.var[:,0,0]
		pi_k = ik.pi

	else:
		mu_k = x[np.random.randint(0,x.size,size=k)] # Pick random mu_k
		var_k = np.repeat(np.var(x),k)
		pi_k = np.repeat(1./k,k)
	theta = _results_gmm(k,pi_k,None,mu_k,var_k,None)
	
	iteration = 0
	while iteration < maxiter:
		# E step
		r = theta.pi[None,:]*Nk_gaussian(x,theta.mu,theta.var)
		theta.r = r/(r.sum(1)[:,None]+1e-300)
		
		# M step
		n = np.sum(theta.r,axis=0)
		theta.mu = 1./n * np.sum(theta.r*x[:,None],axis=0)
		theta.var= 1./n * np.sum(theta.r *(x[:,None]-theta.mu[None,:])**2.,axis=0)
		theta.pi = n / n.sum()
		
		# Compute log-likelihood
		theta.ll_last = theta.ll
		theta.ll = np.sum(np.log(np.sum(theta.pi[None,:] * Nk_gaussian(x,theta.mu,theta.var),axis=-1)))

		# Check convergence
		if np.abs((theta.ll - theta.ll_last)/theta.ll_last) < relative_threshold:
			break
		iteration += 1
	theta.sort()
	return theta