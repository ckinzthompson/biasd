################# Versions#########
# Anaconda		- 2.3.0
## Python		- 2.7.10
### Numpy		- 1.9.2
### Scipy		- 0.16.0
##################################

import numpy as np
np.set_printoptions(precision=4,linewidth=180)
np.seterr(all='ignore')
eps = np.finfo(float).eps
from scipy import special
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import minimize

from os import path
from sys import stdout
import multiprocessing as mp
import cPickle as pickle
import time


def pdot(a,b):
	""" 
	Takes dot product along last two dimensions:
	i.e., dot((N,K,D,D),(N,K,D,D)) --> (N,K,1,1)
	"""
	return np.einsum('...ij,...jk->...ik',a,b)

class stats:
	"""
	Class containing statistical functions for:
	Uniform, Normal, Beta, Gamma, and Wishart distributions
	"""
	
	#Probability distribution functions (PDFs) with parameter support.
	@staticmethod
	def p_uniform(x,a,b):
		return np.nan_to_num(1./(b-a)) * (x >= a) * (x <= b)
	@staticmethod
	def p_gauss(x,mu,std):
		return (1./(std*np.sqrt(2.*np.pi))*np.exp(-.5/std/std*(x-mu)**2.))
	@staticmethod
	def p_beta(x,a,b):
		return np.nan_to_num(np.exp((a-1.)*np.log(x)+(b-1.)*np.log(1.-x) - special.betaln(a,b))) * (x > 0) * (x < 1)
	@staticmethod
	def p_gamma(x,a,b):
		return np.nan_to_num(np.exp(a*np.log(b) +  (a-1.)*np.log(x) + (-b * x) - special.gammaln(a))) * (x > 0)
	
	#Functions that calculate the expectation values E[x] of certain PDFs
	@staticmethod
	def mean_uniform(a,b):
		return .5 *(a+b)
	@staticmethod
	def mean_gauss(a,b):
		return a
	@staticmethod
	def mean_beta(a,b):
		return a/(a+b)
	@staticmethod
	def mean_gamma(a,b):
		return a/b
	
	#Functions that calculate the variances (E[x^2] - E[x]^2) of certain PDFs
	@staticmethod
	def var_uniform(a,b):
		return 1./12. *(b-a)**2.
	@staticmethod
	def var_gauss(a,b):
		return b**2.
	@staticmethod
	def var_beta(a,b):
		return a*b/((a*b)**2.*(a+b+1.))
	@staticmethod
	def var_gamma(a,b):
		return a/b/b
	
	@staticmethod
	def moments_to_params(disttype,first,second):
		"""
		Calculates the distribution parameters (e.g., alpha, & beta) from E[x] and E[x^2].
		This helps for moment matching
		"""
		variance = second - first**2.
		if variance > 0:
			if disttype == "beta":
				alpha = first*(first*(1.-first)/variance-1.)
				beta = (1.-first)*(first*(1.-first)/variance-1.)
			elif disttype == "gamma":
				alpha = first*first/variance
				beta = first/variance
			else:
				alpha = first
				beta = np.sqrt(variance)
			return np.array([alpha,beta])
		return np.array((np.nan,np.nan))
	
	@staticmethod
	def rv_multivariate_normal(mu,cov,number=1):
		"""
		Generate random numbers from a multivariate normal distribution
		"""
		#Calculate transformation to skew symmetric variates to desired shape
		l = np.linalg.cholesky(cov)
		#Draw symmetric normally distributed random numbers for each dimension
		x = np.random.normal(size=(number,np.size(mu)))
		#Transform by shifting to the mean, and skew according to covariance.
		return mu[None,:] + np.dot(x,l.T)

	#Wishart distribution values (see C. Bishop - Pattern Recognition and Machine Learning)
	#Calculate values for last axis, i.e., they are vectorized to nu ~ (...,d) and W ~ (...,d,d)
	@staticmethod
	def wishart_ln_B(W,nu):
		d = W.shape[-1]
		return -nu/2.*np.log(np.linalg.det(W)) - nu*d*np.log(2.)/2. - d*(d-1.)/4.*np.log(np.pi) - special.gammaln((nu[...,None] + 1 -np.linspace(1,d,d).reshape((1,)*nu.ndim+(d,)))/2.).sum(-1)
	@staticmethod
	def wishart_E_ln_det_lam(W,nu):
		d = W.shape[-1]
		return d*np.log(2.) + np.log(np.linalg.det(W)) + special.psi((nu[...,None] + 1 -np.linspace(1,d,d).reshape((1,)*nu.ndim+(d,)))/2.).sum(-1)
	@staticmethod
	def wishart_entropy(W,nu):
		d = W.shape[-1]
		return -stats.wishart_ln_B(W,nu) - (nu - d - 1.)/2. * stats.wishart_E_ln_det_lam(W,nu) + nu*d/2.


class dist:
	"""
	Represents a probability distribution
	distribution is a string for the name of the distribution
	p1, and p2 are the parameters for that distribution
	"""
	
	def __init__(self,distribution,p1,p2):
		self.type = distribution.lower()
		self.p1 = p1
		self.p2 = p2
		
		#Make distribution name types uniformily stored
		if self.type in ['uniform','u',0]:
			self.type = 'uniform'
		elif self.type in ['normal','norm','n','gaussian','gauss',1]:
			self.type = 'normal'
		elif self.type in ['beta','b',2]:
			self.type = 'beta'
		elif self.type in ['gamma','g',3]:
			self.type = 'gamma'
			
		#Check to see if this makes a valid distribution
		self.good = self.good_check()

	def good_check(self):
		"""
		Check if parameters are withing support range and return 1 if so
		"""
		if self.type == 'uniform':
			if np.isfinite(self.p1) and np.isfinite(self.p2) and self.p1 < self.p2:
				return 1
		elif self.type == 'normal':
			if np.isfinite(self.p1) and self.p2 > 0.:
				return 1
		elif self.type == 'beta' or self.type == 'gamma':	
			if self.p1 > 0. and self.p2 > 0.:
				return 1
		return 0
	
	def pdf(self,x):
		"""
		Return the probability distribution function
		x can be a vector
		"""
		if self.good:
			if self.type == "uniform":
				pdffxn = stats.p_uniform(x,self.p1,self.p2)
			elif self.type == "normal":
				pdffxn = stats.p_gauss(x,self.p1,self.p2)
			elif self.type == "beta":
				pdffxn = stats.p_beta(x,self.p1,self.p2)
			elif self.type == "gamma":
				pdffxn = stats.p_gamma(x,self.p1,self.p2)
			return pdffxn
		return 0
	
	def logpdf(self,x):
		"""
		Return the log of the probability distribution function
		"""
		return np.log(self.pdf(x))
	
	def mean(self):
		"""
		Calculate E[x]
		"""
		if self.type == "uniform":
			mean = stats.mean_uniform(self.p1,self.p2)
		elif self.type == "normal":
			mean = stats.mean_gauss(self.p1,self.p2)
		elif self.type == "beta":
			mean = stats.mean_beta(self.p1,self.p2)
		elif self.type == "gamma":
			mean = stats.mean_gamma(self.p1,self.p2)
		return mean
		
	def var(self):
		"""
		Calculate E[x^2] - E[x]^2
		"""
		if self.type == "uniform":
			var = stats.var_uniform(self.p1,self.p2)
		elif self.type == "normal":
			var = stats.var_gauss(self.p1,self.p2)
		elif self.type == "beta":
			var = stats.var_beta(self.p1,self.p2)
		elif self.type == "gamma":
			var = stats.var_gamma(self.p1,self.p2)
		return var
		
	def random(self,size_rvs):
		"""
		Generate random numbers in shape of size_rvs
		"""
		#At least correct for numpy 1.9.2
		np.random.seed()
		if self.type == "uniform":
			rvs = np.random.uniform(self.p1,self.p2,size_rvs)
		elif self.type == "normal":
			rvs = np.random.normal(self.p1,self.p2,size_rvs)
		elif self.type == "beta":
			rvs = np.random.beta(self.p1,self.p2,size_rvs)
		elif self.type == "gamma":
			rvs = np.random.gamma(shape=self.p1,scale=1./self.p2,size=size_rvs)
		return rvs

class biasddistribution:
	"""
	Stores the five parameter probability distribution functions used for the BIASD \Theta.
	\Theta = [\epsilon_1, \epsilon_2, \sigma, \k_1, \k_2]
	This is used for both the prior and the posterior probability distributions
	"""
	def __init__(self,e1,e2,sigma,k1,k2):
		self.names = ['e1','e2','sigma','k1','k2']
		self.e1 = e1
		self.e2 = e2
		self.sigma = sigma
		self.k1 = k1
		self.k2 = k2
		self.list = [self.e1,self.e2,self.sigma,self.k1,self.k2]
		
		#Ensure each distribution in \Theta is sound
		self.complete = self.test_distributions()
		if self.complete != 1:
			print "Distributions are incomplete"
		
	def test_distributions(self):
		"""
		If all distributions in \Theta are correct and return 1
		"""
		good = 1
		for dists in self.list:
			try:
				if dists.good != 1:
					good = 0
					
			except:
				good = 0
		return good
		
	def get_dist_means(self):
		"""
		Calculate means of \Theta
		"""
		if self.complete == 1:
			return np.array((self.e1.mean(),self.e2.mean(),self.sigma.mean(),self.k1.mean(),self.k2.mean()))
		else:
			print "Distributons are incomplete"
			return np.repeat(np.nan,5)
	
	def get_dist_vars(self):
		"""
		Calculate the variances of \Theta
		"""
		if self.complete == 1:
			return np.array((self.e1.var(),self.e2.var(),self.sigma.var(),self.k1.var(),self.k2.var()))
		else:
			print "Distributons are incomplete"
			return np.repeat(np.nan,5)
			
	def sum_log_pdf(self,theta):
		"""
		Returns \Sum_i \ln p\left( \theta_i \right) evaluated at theta (list of numpy array)
		"""
		if self.complete == 1:
			ll = 0
			for theta_i,distribution in zip(theta,self.list):
				ll += np.log(distribution.pdf(theta_i))
			return ll
		else:
			print "Distributions are incomplete"
			return np.nan
			
	def which_bad(self):
		"""
		Figure out which of the distributions is bad
		"""
		if self.complete != 1:
			print "Bad Distributions:"
			baddists=[]
			for i,j in zip([dists.good for dists in self.list],self.names):
				if i != 1:
					print j
					baddists.append(j)
			return baddists
		else:
			print "All distributions seem complete"
			return None
			
	def random_theta(self):
		"""
		Generate a random [\epsilon_1, \epsion_2, \sigma, k_1, k_2] from the BIASD distributions
		"""
		theta = np.repeat(np.nan,5)
		if self.complete == 1:
			#Try a max of 100 times
			for i in range(100):
				for j,distribution in zip(range(5),self.list):
					theta[j] = distribution.random(1)
				#Enforce conditions \epsilon_1 < \epsilon_2, and others are > 0
				if theta[0] < theta[1] and theta[2] > 0. and theta[3] > 0. and theta[4] > 0.:
					break
			theta = np.repeat(np.nan,5)
		return theta

def python_integrand(x,d,e1,e2,sigma,k1,k2,tau):	
	if x < 0. or x > 1. or k1 <= 0. or k2 <= 0. or sigma <= 0. or tau <= 0. or e1 >= e2:
		return 0.
	else:
		k = k1 + k2
		p1 = k2/k
		p2 = k1 /k
		y = 2.*k*tau * np.sqrt(p1*p2*x*(1.-x))
		z = p2*x + p1*(1.-x)
		pf = 2.*k*tau*p1*p2*(special.i0(y)+k*tau*(1.-z)*special.i1(y)/y)*np.exp(-z*k*tau)
		py = 1./np.sqrt(2.*np.pi*sigma**2.)*np.exp(-.5/sigma/sigma*(d-(e1*x+e2*(1.-x)))**2.) * pf
		return py

def python_integral(d,e1,e2,sigma,k1,k2,tau):
	return quad(python_integrand,0.,1.,args=(d,e1,e2,sigma,k1,k2,tau),limit=1000)[0]

integral = python_integral
integral = np.vectorize(integral)

def load_c_integral(integrandpath):
	global integral
	# integrandpath = '/home/colin/Documents/data/20150821_biasd/integrand_gsl.so' #34.6x increase
	# integrandpath = '/home/colin/Documents/data/20150821_biasd/integrand_full_gsl.so' #31.8x increase
	# integrandpath = '/home/colin/Documents/data/20150821_biasd/integrand_cephes.so' #23.7x increase
	if path.isfile(integrandpath):
		import ctypes
		lib = ctypes.CDLL(integrandpath)
		integrand = lib.integrand
		integrand.restype = ctypes.c_double
		integrand.argtypes = (ctypes.c_int,ctypes.c_double)
		def c_integral(d,e1,e2,sigma,k1,k2,tau):
			return quad(integrand,0.,1.,args=(d,e1,e2,sigma,k1,k2,tau))[0]
		integral = c_integral
		print "Loaded integrand written in C"
		integral = np.vectorize(integral)
		return 1
	else:
		print "Couldn't find the compiled library"
		print "Using integrand written in Python"
		return 0

def calc_hessian(fxn,x,eps = np.sqrt(np.finfo(np.float64).eps)):
	
	#Using Abramowitz & Stegun Eqn. 25.3.23, and 25.3.26
	h = np.zeros((x.size,x.size))
	y00 = fxn(x)
	
	for i in range(x.size):
		for j in range(x.size):
			if j < i:
				h[i,j] = h[j,i]
			else:
				if i == j:
					x10 = x.copy()
					xm10 = x.copy()
					
					x10[i] += eps
					xm10[i] -= eps
					
					y10 = fxn(x10)
					ym10 = fxn(xm10)
					
					h[i,j] = eps**(-2.) * (y10 - 2.*y00 + ym10)
					
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

def switch_to_python_integral():
	global integral
	integral = python_integral
	integral = np.vectorize(integral)
	print "Using integrand written in Python"

def log_likelihood(theta,d,tau):
	global integral
	e1,e2,sigma,k1,k2 = theta
	p1 = k2/(k1+k2)
	p2 = 1.-p1
	out = integral(d,e1,e2,sigma,k1,k2,tau)
	peak1 = stats.p_gauss(d,e1,sigma)
	peak2 = stats.p_gauss(d,e2,sigma)
	out += p1*peak1*np.exp(-k1*tau)
	out += p2*peak2*np.exp(-k2*tau)
	return np.log(out+eps)

def log_posterior(d,priors,tau,theta):
	return log_likelihood(theta,d,tau).sum()+priors.sum_log_pdf(theta)

def test_speed(n):
	d = np.linspace(-2,1.2,5000)
	t0 = time.time()
	for i in range(n):
		# quad(integrand,0.,1.,args=(.1,0.,1.,.05,3.,8.,.1))[0]
		y = log_likelihood([0.,1.,.05,3.,8.],d,.1).sum()
	t1 = time.time()
	print "Total time for "+str(n)+" runs: ",np.around(t1-t0,4)," (s)"
	print 'Average speed: ', np.around((t1-t0)/n/d.size*1.e6,4),' (usec/datapoint)'

def variational_gmm(x,nstates,maxiter=5000,lowerbound_threshold=1e-16):
################# Bishop Algorithim - Chapter 10, Section 2


######## Initialize
	ntraces = x.shape[0]
	ndim = x.shape[1]

	alpha_0 = .1
	beta_0 = 1e-20
	nu_0 = ndim + 1.
	W_0 = np.identity(ndim,dtype='f')
	Winv_0 = np.linalg.inv(W_0)
	m_0 = np.zeros((ndim),dtype='f')
	
	np.random.seed()
	r_nk = np.array([np.random.dirichlet(np.repeat(alpha_0,nstates)) for _ in range(ntraces)])
	N_k = np.zeros((nstates))
	m_k = np.repeat(m_0[None,:],nstates,axis=0)
	W_k = np.repeat(W_0[None,:,:],nstates,axis=0)
	alpha_k = np.repeat(alpha_0,nstates)
	
	lb = None
	state_log = None
	finished_counter = 0

	E_lam = np.zeros((nstates))
	E_pi = np.zeros((nstates))
	E_mulam = np.zeros((ntraces,nstates))

	def calc_lowerbound(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,E_lam,E_pi,E_mulam):
		eq71 = 0.5 * np.sum( N_k * (E_lam - ndim/beta_k - nu_k * np.trace(pdot(S_k,W_k),axis1=-2,axis2=-1) -nu_k*pdot((xbar_k-m_k)[:,None,:],pdot(W_k,(xbar_k-m_k)[:,:,None]))[:,0,0]  - ndim*np.log(2.*np.pi) ))
		eq72 = np.sum(np.sum(r_nk*E_pi[None,:],axis=1),axis=0)
		eq73 = special.gammaln(alpha_0*nstates) - nstates*special.gammaln(alpha_0) + (alpha_0-1.)*np.sum(E_pi)
		eq74 = 0.5 * np.sum( ndim*np.log(beta_0/(2.*np.pi)) + E_lam - ndim*beta_0/beta_k - beta_0 * nu_k * pdot((m_k-m_0)[:,None,:],pdot(W_k,(m_k-m_0)[:,:,None])))
		eq74 += nstates * stats.wishart_ln_B(W_0,np.array((nu_0))) + (nu_0 - ndim -1.)/2. * np.sum(E_lam) - 0.5 * np.sum(nu_k * np.trace(pdot(Winv_0[None,...],W_k),axis1=-2,axis2=-1))
		eq75 = np.sum(np.sum(r_nk*np.log(r_nk+1e-300),axis=1),axis=0)
		eq76 = np.sum((alpha_k-1.)*E_pi) + special.gammaln(alpha_k.sum()) - np.sum(special.gammaln(alpha_k))
		eq77 = 0.5 *np.sum(E_lam + ndim*np.log(beta_k/(2.*np.pi)) - ndim - 2.*stats.wishart_entropy(W_k,nu_k))
		lowerbound =  np.nan_to_num((eq71,eq72,eq73,eq74,-eq75,-eq76,-eq77)).sum()
		return lowerbound#,[eq71,eq72,eq73,eq74,eq75,eq76,eq77]

	it = 0
	while 1:
		if it > maxiter or finished_counter > 5:
			break
		
############ M-Step
		N_k = np.sum(r_nk,axis=0)
		xbar_k = np.sum(r_nk[:,:,None]*x[:,None,:],axis=0)/(N_k+1e-16)[:,None]
		S_k = np.sum(r_nk[:,:,None,None]*pdot((x[:,None,:] - xbar_k[None,:,:])[:,:,:,None],(x[:,None,:] - xbar_k[None,:,:])[:,:,None,:]),axis=0)/(N_k+1e-16)[:,None,None]

		nu_k = nu_0+ N_k
		beta_k = beta_0 + N_k
		alpha_k = alpha_0 + N_k
		m_k = (beta_0*m_0 + N_k[:,None]*xbar_k)/beta_k[:,None]
		W_k = np.linalg.inv( Winv_0 + N_k[:,None,None]*S_k + (beta_0*N_k/(beta_0 + N_k))[:,None,None] * pdot((xbar_k - m_0)[:,:,None],(xbar_k - m_0)[:,None,:]) )
		
############ E-Step
		E_lam = np.sum(special.psi((nu_k[:,None] + 1. - np.linspace(1,ndim,ndim)[None,:])/2.),axis=1) + ndim*np.log(2.) + np.log(np.linalg.det(W_k))
		E_pi = special.psi(alpha_k) - special.psi(alpha_k.sum())
		E_mulam = ndim/beta_k[None,:] + nu_k[None,:] * pdot((x[:,None,None,:] - m_k[None,:,None,:]),pdot(W_k[None,:,:,:],(x[:,None,:,None] - m_k[None,:,:,None])))[:,:,0,0]

		rho_nk = E_pi[None,:] + .5 * E_lam[None,:] - ndim/2.*np.log(2.*np.pi) - .5*E_mulam
		rho_nk -= rho_nk.max(1)[:,None]
		rho_nk = np.exp(rho_nk)
		r_nk = rho_nk/np.sum(rho_nk,axis=1)[:,None]

#~ ############ Remove Unpopulated States
		#~ cutk = np.nonzero(N_k < alpha_0*1e-6)[0]
		#~ if cutk.size > 0:
			#~ nstates -= cutk.size
			#~ if not np.ndim(state_log):
				#~ state_log = np.array([it,nstates])[None,:]
			#~ else:
				#~ state_log = np.append(state_log,np.array([it,nstates])[None,:],axis=0)
			#~ xbar_k = np.delete(xbar_k,cutk,axis=0)
			#~ S_k = np.delete(S_k,cutk,axis=0)
			#~ N_k = np.delete(N_k,cutk,axis=0)
			#~ rho_nk = np.delete(rho_nk,cutk,axis=1)
			#~ r_nk = np.delete(r_nk,cutk,axis=1)
			#~ nu_k = np.delete(nu_k,cutk,axis=0)
			#~ beta_k = np.delete(beta_k,cutk,axis=0)
			#~ alpha_k = np.delete(alpha_k,cutk,axis=0)
			#~ m_k = np.delete(m_k,cutk,axis=0)
			#~ W_k = np.delete(W_k,cutk,axis=0)
			#~ E_lam = np.delete(E_lam,cutk,axis=0)
			#~ E_pi = np.delete(E_pi,cutk,axis=0)
			#~ E_mulam = np.delete(E_mulam,cutk,axis=1)

############ Lowerbound Threshold for Convergence
		l = calc_lowerbound(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,E_lam,E_pi,E_mulam)
		
		if not np.ndim(lb):
			lb = np.array((it,l))[None,:]
		else:
			lb = np.append(lb,np.array((it,l))[None,:],axis=0)
		
		#print it,nstates,l,-lb[it-1,1]+lb[it,1]
		#~ if it > 1 and (np.abs((lb[it,1]-lb[it-1,1])/lb[it,1]) < lowerbound_threshold ):# or lb[it,1] < lb[it-1,1]):
			#~ finished_counter += 1
		#~ else:
			#~ finished_counter = 0
		if it > 1 and lb[it-1,1]==lb[it,1]:
			break
			#~ finished_counter += 1
		#~ else:
			#~ finished_counter = 0
			
		it += 1
	
	#return [alpha_k,r_nk,m_k,beta_k,nu_k,W_k,S_k,lb,state_log]
	xsort = alpha_k.argsort()[::-1]
	return [alpha_k[xsort],r_nk[:,xsort],m_k[xsort],beta_k[xsort],nu_k[xsort],W_k[xsort],S_k[xsort],lb,state_log]

class Laplace_Worker(mp.Process):
	def __init__(self,queue_in,queue_out):
		self.__queue_in = queue_in
		self.__queue_out = queue_out
		mp.Process.__init__(self)
	
	def run(self):
		while 1:
			item = self.__queue_in.get()
			if item is None:
				self.__queue_out.put(item)
				break
			self.__queue_out.put(item.laplace_approximation())

class Variational_Worker(mp.Process):
	def __init__(self,queue_in,queue_out):
		self.__queue_in = queue_in
		self.__queue_out = queue_out
		mp.Process.__init__(self)
	
	def run(self):
		while 1:
			item = self.__queue_in.get()
			if item is None:
				self.__queue_out.put(item)
				break
			self.__queue_out.put(variational_gmm(*item))

class Watcher(mp.Process):
	def __init__(self,queue_in,queue_out,numworkers,numjobs,batchnum):
		self.__queue_out = queue_out
		self.__queue_in = queue_in
		self.numworkers = numworkers
		self.numjobs = numjobs
		self.batchnum = batchnum
		mp.Process.__init__(self)
		
	def run(self):
		pillcount = 0
		self.resultcount = 0
		#stdout.write("\nBatch "+str(self.batchnum)+": 000.00%")
		stdout.write("\nBatch "+str(self.batchnum+1)+": "+"_"*self.numjobs)
		stdout.flush()
		
		while pillcount < self.numworkers:
			item = self.__queue_in.get()
			if item is None:
				pillcount += 1
			else:
				self.__queue_out.put(item)
				self.resultcount += 1
				#stdout.write("\b"*7+'{:06.2f}'.format(self.resultcount/float(self.numjobs)*100.) + "%")
				stdout.write("\b"*self.numjobs+"X"*self.resultcount+"_"*(self.numjobs-self.resultcount))
				stdout.flush()

class laplace_posterior:
	def __init__(self,means,covars):
		self.mu = means
		self.covar = covars

class ensemble:
	def __init__(self,x,params):
		self.x = x
		self.alpha,self.r,self.m,self.beta,self.nu,self.W,self.S_k,self.lowerbound,self.state_log = params
		self.var = np.zeros_like(self.m)
		self.covar = np.zeros_like(self.W)
		for i in range(np.size(self.alpha)):
			self.covar[i] = np.linalg.inv(self.nu[i]*self.W[i])
			self.var[i] = np.diag(self.covar[i])
		self.z = None
	
	def get_rvs(self,state,n):
		rvs = np.zeros((n,5))
		for i in range(5):
			rvs[:,i] = np.random.normal(self.m[state,i],self.var[state,i]**.5,size=n)
			if i  > 1:
				while 1:
					xbad = rvs[:,i] <= 0.
					if np.any(xbad):
						rvs[np.nonzero(xbad)[0],i] = np.random.normal(self.m[state,i],self.var[state,i]**.5,size=xbad.sum())
					else:
						break
		return rvs
		#~ return np.random.multivariate_normal(self.m[state],self.S_k[state],size=n)
	
	def report(self):
		rep = "Iterations Lowerbound \n"+str(self.lowerbound[-1][0]) +"  "+str(self.lowerbound[-1][1]) + "\n\n"
		rep += "Final States\n" + str(self.alpha.size)+"\n\n"
		rep += "Fraction\nState Mean Var.\n" 
		states = np.size(self.alpha)
		a0 = self.alpha.sum()
		for i in range(states):
			rep += str(i+1)+" "+ str(self.alpha[i]/a0) + " " + str(self.alpha[i]*(a0-self.alpha[i])/(a0**2.*(a0+1.))) + "\n"
		rep += "\n"
		
		theta = ['E1','E2','Sigma','K1','K2']
		for i in range(5):
			rep += "\n"+ theta[i] +"\n"+ "State Mean Var.\n"
			for j in range(states):
				rep += str(j+1) + " " + str(self.m[j,i]) + " " + str(self.var[j,i]) + "\n"
		return rep

class trace:
	def __init__(self, data, tau=None, prior=None,identity=0):
		self.data = data.flatten()
		self.tau = float(tau)
		self.prior = prior
		self.posterior = None
		self.identity = identity
	
	def log_likelihood(self,theta):
		return log_likelihood(theta,self.data,self.tau)

	def log_posterior(self,theta):
		return log_posterior(self.data,self.prior,self.tau,theta)
	
	def find_map(self,meth='nelder-mead',xx=None,nrestarts=5):
		ylist = []
		if type(xx).__name__ != 'ndarray':
			xx = self.prior.get_dist_means()
		ylist.append(minimize(lambda theta: -1.*self.log_posterior(theta),x0=xx,method=meth))
		for i in range(nrestarts):
			xx = self.prior.random_theta()
			ylist.append(minimize(lambda theta: -1.*self.log_posterior(theta),x0=xx,method=meth))
		ymin = np.inf
		for i in ylist:
			if i['success']:
				if i['fun'] < ymin:
					ymin = i['fun']
					y = i
		if ymin == np.inf:
			y = None
		return y

	def laplace_approximation(self):
		def sanitize_uniform(x,p):
			for i in range(5):
				if p.list[i].type == 'uniform':
					if np.abs(p.list[i].p1 - x[i]) < 1e-5:
						x[i] += 1e-5
					elif np.abs(p.list[i].p2 - x[i]) < 1e-5:
						x[i] -= 1e-5
			return x
					
		mind = self.find_map()
		if not mind is None:
			if mind['success']: 
				mu = sanitize_uniform(mind['x'],self.prior)
				feps = np.sqrt(np.finfo(np.float).eps)
				hessian = calc_hessian(self.log_posterior,mu,eps=feps)
				try:
					while np.any(np.linalg.eig(-hessian)[0] <= 0.):
						feps *= 2.
						hessian = calc_hessian(self.log_posterior,mu,eps=feps)
					var = np.linalg.inv(-hessian)
					#Ensure symmetry if witin machine error
					if np.allclose(var,var.T):
						var = np.tri(5,5,-1)*var+(np.tri(5,5)*var).T
						return (self.identity,laplace_posterior(mu,var))
				except np.linalg.LinAlgError:
					return (self.identity,None)
		return (self.identity,None)

class dataset:
	def __init__(self,data_fname=None,fmt='2D-NxT',tau=None,temperature = 25., title = None, prior = None,analysis_fname=None):
		self.data_fname = data_fname
		self.fmt = fmt
		self.tau =tau
		self.temperature = temperature
		self.title = title
		self.prior = prior
		self.traces = []
		self.analysis_fname = analysis_fname
		self.ensemble_result = None
	
	@staticmethod
	def _convert_2D_to_1D(trace_matrix):
		"Convert NxT (with NaN\'s of inf\'s for no data) to 2x(N*T) with labels format"
		if trace_matrix.ndim == 1:
			trace_matrix = trace_matrix[None,:]
		identities = np.array([])
		traces = np.array([])
		for i in range(trace_matrix.shape[0]):
			l = np.isfinite(trace_matrix[i]).sum()
			identities = np.append(identities,np.repeat(i,l))
			traces = np.append(traces,trace_matrix[i,:l])
		return np.array((identities,traces))

	@staticmethod
	def _convert_1D_to_2D(trace_matrix):
		"Convert 2x(N*T) with labels format to NxT format (with NaN\'s for no data)"
	
		ns = np.unique(trace_matrix[0])
		sizes = (trace_matrix[0][None,:] == ns[:,None]).sum(1)
		traces_out = np.ones((ns.size,sizes.max()))
		traces_out[:,:] = np.nan
	
		for i in range(ns.size):
			d = trace_matrix[1][trace_matrix[0] == ns[i]]
			traces_out[i,:sizes[i]] = d
		return traces_out
	
	
	def load_data(self):
		if type(self.data_fname) == str:
			if path.isfile(self.data_fname):
				try:
					self.data = np.loadtxt(self.data_fname)
					if self.fmt == '2D-TxN':
						self.data = self.data.T
					if self.fmt != '1D':
						self.data = self._convert_2D_to_1D(self.data)
					if self.fmt == '1D':
						self.data = self._convert_1D_to_2D(self._convert_2D_to_1D(self.data))
				except:
					self.data = None
					print "Couldn't load "+self.data_fname
					return
			else:
				print self.data_fname + " isn't a file"
				return
			
			ns = np.unique(self.data[0])
			for i in range(ns.size):
				self.traces.append(trace(self.data[1][self.data[0]==ns[i]],tau=self.tau,prior=self.prior,identity=i))
			self.n_traces = len(self.traces)
			
			
	def save_analysis(self):
		if self.analysis_fname:
			f = open(self.analysis_fname,'wb')
			pickle.dump(self.__dict__,f,2)
			f.close()
	
	def load_analysis(self):
		if path.isfile(self.analysis_fname):
			f = open(self.analysis_fname,'rb')
			tmp_dict = pickle.load(f)
			f.close()
			self.__dict__.update(tmp_dict)
		else:
			print "No file called ",self.analysis_fname
	
	def update(self):
		for tracei in self.traces:
			tracei.prior = self.prior
			tracei.tau = self.tau
	
	def run_laplace(self,nproc=1):
		if nproc > mp.cpu_count():
			print "Using max number of CPUs: "+str(mp.cpu_count())
			nproc = mp.cpu_count()
		
		j = 1
		print "-----------------\nLaplace Approximations"
		t0 = time.time()
		if nproc == 1:
			for tracei in self.traces:
				print j,"/",np.size(self.traces)
				item = tracei.laplace_approximation()
				tracei.posterior = item[1]
				j += 1
		else:
			#Multiprocessing has a memory problem, so run in batches
			batchsize = 100
			while batchsize % nproc != 0:
				batchsize -= 1
			batchnum = int(self.n_traces/batchsize)+1

			tracelist= []
			print "Batches - ",batchnum,
			for bi in range(batchnum):
				tracelist = self.traces[bi::batchnum]
				
				queue_work = mp.Queue(nproc)
				queue_out = mp.Queue()
				queue_results = mp.Queue()

				workers = []
				for i in range(nproc):
					worker = Laplace_Worker(queue_work,queue_out)
					worker.start()
					workers.append(worker)
				watcher = Watcher(queue_out,queue_results,nproc,len(tracelist),bi)
				watcher.start()

				for tracei in tracelist:
					queue_work.put(tracei)
				
				#Poison Pills
				for i in workers:
					queue_work.put(None)
				
				#Wait for it
				for worker in workers:
					worker.join()

				for i,n in enumerate(tracelist):
					y = queue_results.get()
					self.traces[y[0]].posterior = y[1]
				watcher.join()
				
		
		t1 = time.time()
		print "\nTime:",t1-t0
		#self.save_analysis()
		
	def variational_ensemble(self,nstates=20,nsamples=100,nrestarts=5,nproc=1):
		if nproc > mp.cpu_count():
			print "Using max number of CPUs: "+str(mp.cpu_count())
			nproc = mp.cpu_count()
			
		self._ensemble_results = []

		x = []
		trace_id = []
		tid = 0
		t0 = time.time()
		
		for tracei in self.traces:
			if not tracei.posterior is None:
				trace_id.append(tid)
				tid += 1
				x.append(stats.rv_multivariate_normal(tracei.posterior.mu,tracei.posterior.covar,number=nsamples))
				
		x = np.array(x)
		
		if x.shape[0] > 0.:
			print x.shape
			print "-----------------\nVariational Mixtures"
			
			t0 = time.time()
			
			if nstates > 1:
				queue_work = mp.Queue(nproc)
				queue_out = mp.Queue()
				queue_results = mp.Queue()
				
				workers = []
				for i in range(nproc):
					worker = Variational_Worker(queue_work,queue_out)
					worker.start()
					workers.append(worker)
				watcher = Watcher(queue_out,queue_results,nproc,1+nrestarts*(nstates-1),0)
				watcher.start()
				
				queue_work.put((x.reshape((x.shape[0]*x.shape[1],5)),1))
				for st in np.linspace(2,nstates,nstates-1,dtype='i'):
					for nr in range(nrestarts):
						queue_work.put((x.reshape((x.shape[0]*x.shape[1],5)),st))
				
				#Poison Pills
				for i in range(nproc):
					queue_work.put(None)
				
				#Wait for it
				for worker in workers:
					worker.join()
				
				print "\n"
				for i in range(1+nrestarts*(nstates-1)):
					y = queue_results.get()
					print "States:",y[0].size,", Iterations:",int(y[-2][-1][0]),", Lowerbound:",y[-2][-1][1]
					self._ensemble_results.append(y)
				watcher.join()
			
			else:
				y = variational_gmm(x.reshape((x.shape[0]*x.shape[1],5)),1)
				self._ensemble_results.append(y)

			print "Post-Processing"
			self._lb_states = np.zeros((3,nstates)) - 1.
			self._lb_states[0] = np.linspace(1,nstates,nstates)
			self._lb_states[1] = np.repeat(-np.inf,self._lb_states[1].size)
			self.ensemble_result = ensemble(x,self._ensemble_results[0])
			for ind,er in enumerate(self._ensemble_results):
				er_lb = er[-2][-1,1]
				er_states = er[0].size
				if er_lb > self.ensemble_result.lowerbound[-1,1]:
					self.ensemble_result = ensemble(x,er)
				if self._lb_states[1,er_states-1] < er_lb:
					self._lb_states[1,er_states-1] = er_lb
					self._lb_states[2,er_states-1] = ind
			
			z = np.repeat(-1.,self.n_traces)
			for i in range(len(trace_id)):
				z[i] = self.ensemble_result.r[i*nsamples:(i+1)*nsamples].sum(0).argmax()
			self.ensemble_result.z = z
			
			for k in range(nstates):
				er = ensemble(self.ensemble_result.x,self._ensemble_results[int(self._lb_states[2,k])])
				if k == 0:
					hy,hx = np.histogram(self.data[1],bins=self.data[1].size**.5,normed=1)
					hxx = 0.5*(hx[1:]+hx[:-1])
					x0 = er.get_rvs(0,1000)
					dx0 = np.zeros_like(hxx)
					for x in x0:
						dx0 += np.exp(log_likelihood(x,hxx,self.tau))
					dx0 /= dx0.sum() * (hxx[1]-hxx[0])
					px0 = dx0*1.
					self._histograms = [[[hx,hy,hxx,px0]]]
				else:
					counts = np.zeros((self.n_traces,er.alpha.size))
					for i in range(len(trace_id)):
						counts[i] = er.r[trace_id[i]*nsamples:(trace_id[i]+1)*nsamples].sum(0)
					hk = []
					for kk in range(k+1):
						q = np.array(())
						w = np.array(())
						for i in self.traces:
							q = np.append(q,i.data)
							w = np.append(w,np.repeat(counts[i.identity,kk],i.data.size))
						chy,chx = np.histogram(q,bins=hx,normed=1,weights=w)
						chy *= counts.sum(0)[kk]/counts.sum()
						try:
							x0 = er.get_rvs(kk,1000)
							dx0 = np.zeros_like(hxx)
							for x in x0:
								dx0 += np.exp(log_likelihood(x,hxx,self.tau))
							dx0 /= dx0.sum() * (hxx[1]-hxx[0])
							px0 = dx0*er.alpha[kk]/er.alpha.sum()
						except:
							px0 = np.zeros_like(hxx)
						hk.append([chx,chy,hxx,px0])
					self._histograms.append(hk)
			
			t1 = time.time()
			print "time:",t1-t0
		else:
			print "No Posteriors?"



prior_personal_distribution = biasddistribution(
dist('normal',0.15,.1),
dist('normal',.75,.1),
dist('gamma',70.,1000.),
dist('Gamma',2.,2./10.),
dist('Gamma',2.,2./10.))

prior_generic_distribution = biasddistribution(
dist('normal',0.,1000.),
dist('normal',1.,1000.),
dist('normal',.1,1000.),
dist('normal',1.,1000.),
dist('normal',1.,1000.))

