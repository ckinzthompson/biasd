import numpy as np
from scipy import special


def plot_cov_ellipse(cov, pos, volume=.5, ax=None, fc='none', ec=[0,0,0], a=1, lw=2):
    """
    Plots an ellipse enclosing *volume* based on the specified covariance
    matrix (*cov*) and location (*pos*). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        volume : The volume inside the ellipse; defaults to 0.5
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
    """

    import numpy as np
    from scipy.stats import chi2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    kwrg = {'facecolor':fc, 'edgecolor':ec, 'alpha':a, 'linewidth':lw}

    # Width and height are "full" widths, not radius
    width, height = 2 * np.sqrt(chi2.ppf(volume,2)) * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, ec='red', **kwrg)

    ax.add_artist(ellip)

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

def calc_lowerbound_old(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,beta_k,nu_k,E_lam,E_pi,E_mulam):
		eq71 = 0.
		eq72 = np.sum(np.sum(r_nk*E_pi[None,:],axis=1),axis=0)
		eq73 = special.gammaln(alpha_0*nstates) - nstates*special.gammaln(alpha_0) + (alpha_0-1.)*np.sum(E_pi)
		eq74 = 0.
		eq75 = np.sum(np.sum(r_nk*np.log(r_nk+1e-16),axis=1),axis=0)
		eq76 = np.sum((alpha_k-1.)*E_pi) + special.gammaln(alpha_k.sum()) - np.sum(special.gammaln(alpha_k))
		eq77 = 0.
		for k in range(nstates):
			eq71 += 0.5 * N_k[k] * (E_lam[k] - ndim/beta_k[k] - nu_k[k]*np.trace(np.dot(S_k[k],W_k[k])) - nu_k[k]*np.dot((xbar_k[k]-m_k[k])[:,None].T,np.dot(W_k[k],(xbar_k[k]-m_k[k])[:,None]))[0][0] - ndim*np.log(2.*np.pi))
			eq74 += 0.5 * (ndim*np.log(beta_0/(2.*np.pi)) + E_lam[k] - ndim*beta_0/beta_k[k] - beta_0*nu_k[k]*np.dot((m_k[k]-m_0)[:,None].T,np.dot(W_k[k],(m_k[k]-m_0)[:,None]))[0][0]) - .5*nu_k[k]*np.trace(np.dot(np.linalg.inv(W_0),W_k[k]))
			H_lamk = - ((-nu_k[k]/2.) * np.log(np.linalg.det(W_k[k])) - (nu_k[k]*ndim/2.)*np.log(2.) - (ndim*(ndim-1.)/4.)*np.log(np.pi) - np.sum(special.gammaln((nu_k[k]+1.-np.linspace(1,ndim,ndim))/2.))) + nu_k[k]*ndim/2. - (nu_k[k] - ndim -1.)/2. * (np.sum(special.psi((nu_k[k]+1.-np.linspace(1,ndim,ndim)))/2.) - ndim * np.log(2.) + np.log(np.linalg.det(W_k[k])))
			eq77 +=.5*E_lam[k] + ndim/2.*np.log(beta_k[k]/(2.*np.pi))-ndim/2. - H_lamk
		eq74 += nstates * ((-nu_0/2.) * np.log(np.linalg.det(W_0)) - (nu_0*ndim/2.)*np.log(2.) - (ndim*(ndim-1.)/4.)*np.log(np.pi) - np.sum(special.gammaln((nu_0+1.-np.linspace(1,ndim,ndim))/2.)))
		eq74 += (nu_0-ndim-1.)/2.*np.sum(E_lam)
		lowerbound = eq71 + eq72 + eq73 + eq74 - eq75 - eq76 - eq77
		#print eq71,eq72,eq73,eq74,eq75,eq76,eq77
		return lowerbound#,[eq71,eq72,eq73,eq74,eq75,eq76,eq77]



def variational_gmm_old(x,nstates,maxiter=500,lowerbound_threshold=1e-6):
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


	it = 0
	while 1:
		if it > maxiter or finished_counter > 1:
			break
		
		####M-Step
		xbar_k = np.zeros((nstates,ndim))
		S_k = np.zeros((nstates,ndim,ndim))
		
		N_k = np.sum(r_nk,axis=0)
		for k in range(nstates):
			for n in range(ntraces):
				xbar_k[k] += r_nk[n,k]*x[n]
			xbar_k[k] /= (N_k[k]+1e-16)
			for n in range(ntraces):
				S_k[k] += r_nk[n,k]*np.dot((x[n]-xbar_k[k])[:,None],(x[n]-xbar_k[k])[:,None].T)
			S_k[k] /= (N_k[k]+1e-16)

		nu_k = nu_0+ N_k
		beta_k = beta_0 + N_k
		for k in range(nstates):
			m_k[k] = (beta_0*m_0 + N_k[k]*xbar_k[k])/beta_k[k]
			W_k[k] = np.linalg.inv( Winv_0 + N_k[k]*S_k[k] + beta_0*N_k[0]/(beta_0 + N_k[k]) * np.dot((xbar_k[k] - m_0)[:,None],(xbar_k[k] - m_0)[:,None].T))
			alpha_k[k] = alpha_0 + N_k[k]
		
		
		####E-Step
		for k in range(nstates):
			E_lam[k] = np.sum(special.psi((nu_k[k] + 1. - np.linspace(1,ndim,ndim))/2.)) + ndim*np.log(2.) + np.log(np.linalg.det(W_k[k]))
			E_pi[k] = special.psi(alpha_k[k]) - special.psi(alpha_k.sum())
			for n in range(ntraces):
				E_mulam[n,k] = ndim/beta_k[k] + nu_k[k] * np.dot((x[n] - m_k[k]).T,np.dot(W_k[k],(x[n] - m_k[k])))
		rho_nk = E_pi[None,:] + .5 * E_lam[None,:] - ndim/2.*np.log(2.*np.pi) - .5*E_mulam
		rho_nk -= rho_nk.max(1)[:,None]
		rho_nk = np.exp(rho_nk)
		r_nk = rho_nk/(1e-16+rho_nk.sum(1)[:,None])
		
		l = lbfxn(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,beta_k,nu_k,E_lam,E_pi,E_mulam)
		
		if not np.ndim(lb):
			lb = np.array((it,l))[None,:]
		else:
			lb = np.append(lb,np.array((it,l))[None,:],axis=0)

		#print it,nstates,l
		if it > 1 and (np.abs((lb[it,1]-lb[it-1,1])/lb[it,1]) < lowerbound_threshold ):# or lb[it,1] < lb[it-1,1]):
			finished_counter += 1
			
		it += 1
		
	xsort = alpha_k.argsort()[::-1]
	return [alpha_k[xsort],r_nk[:,xsort],m_k[xsort],beta_k[xsort],nu_k[xsort],W_k[xsort],S_k[xsort],lb,state_log]

def variational_gmm_fast(x,nstates,maxiter=5000,lowerbound_threshold=1e-16):
	"""
	Variational Gaussian Mixture Model from C. Bishop - Chapter 10, Section 2.
	x is an n by d array, where n is the number of points, and d is the dimensionality of the points.
	maxiter is the maximum number of rounds before stopping if the lowerbound_thershold is not met.
	
	This version is fully vectorized. Possible typo in the lower_bound calculation.
	"""
	#### Initialize
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

	it = 0
	while 1:
		if it > maxiter or finished_counter > 5:
			break
		
		N_k = np.sum(r_nk,axis=0)
		xbar_k = np.sum(r_nk[:,:,None]*x[:,None,:],axis=0)/(N_k+1e-16)[:,None]
		S_k = np.sum(r_nk[:,:,None,None]*pdot((x[:,None,:] - xbar_k[None,:,:])[:,:,:,None],(x[:,None,:] - xbar_k[None,:,:])[:,:,None,:]),axis=0)/(N_k+1e-16)[:,None,None]

		nu_k = nu_0+ N_k
		beta_k = beta_0 + N_k
		alpha_k = alpha_0 + N_k
		m_k = (beta_0*m_0 + N_k[:,None]*xbar_k)/beta_k[:,None]
		W_k = np.linalg.inv( Winv_0 + N_k[:,None,None]*S_k + (beta_0*N_k/(beta_0 + N_k))[:,None,None] * pdot((xbar_k - m_0)[:,:,None],(xbar_k - m_0)[:,None,:]) )
		
		E_lam = np.sum(special.psi((nu_k[:,None] + 1. - np.linspace(1,ndim,ndim)[None,:])/2.),axis=1) + ndim*np.log(2.) + np.log(np.linalg.det(W_k))
		E_pi = special.psi(alpha_k) - special.psi(alpha_k.sum())
		E_mulam = ndim/beta_k[None,:] + nu_k[None,:] * pdot((x[:,None,None,:] - m_k[None,:,None,:]),pdot(W_k[None,:,:,:],(x[:,None,:,None] - m_k[None,:,:,None])))[:,:,0,0]

		rho_nk = E_pi[None,:] + .5 * E_lam[None,:] - ndim/2.*np.log(2.*np.pi) - .5*E_mulam
		rho_nk -= rho_nk.max(1)[:,None]
		rho_nk = np.exp(rho_nk)
		r_nk = rho_nk/np.sum(rho_nk,axis=1)[:,None]


		l = lbfxn(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,beta_k,nu_k,E_lam,E_pi,E_mulam)
		
		if not np.ndim(lb):
			lb = np.array((it,l))[None,:]
		else:
			lb = np.append(lb,np.array((it,l))[None,:],axis=0)
		
		if it > 1 and lb[it-1,1]==lb[it,1]:
			break
			#~ finished_counter += 1
		#~ else:
			#~ finished_counter = 0
			
		it += 1
	
	xsort = alpha_k.argsort()[::-1]
	return [alpha_k[xsort],r_nk[:,xsort],m_k[xsort],beta_k[xsort],nu_k[xsort],W_k[xsort],S_k[xsort],lb,state_log]

def variational_gmm_new(x,nstates,maxiter=5000,lowerbound_threshold=1e-10):
	"""
	Variational Gaussian Mixture Model from C. Bishop - Chapter 10, Section 2.
	x is an n by d array, where n is the number of points, and d is the dimensionality of the points.
	maxiter is the maximum number of rounds before stopping if the lowerbound_thershold is not met.
	
	This version is fully vectorized. Possible typo in the lower_bound calculation.
	"""
	def calc_lowerbound(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,beta_k,nu_k,E_lam,E_pi,E_mulam):
		eq71 = 0.5 * np.sum( N_k * (E_lam - ndim/beta_k - nu_k * np.trace(pdot(S_k,W_k),axis1=-2,axis2=-1) -nu_k*pdot((xbar_k-m_k)[:,None,:],pdot(W_k,(xbar_k-m_k)[:,:,None]))[:,0,0]  - ndim*np.log(2.*np.pi) ))
		eq72 = np.sum(np.sum(r_nk*E_pi[None,:],axis=1),axis=0)
		eq73 = special.gammaln(alpha_0*nstates) - nstates*special.gammaln(alpha_0) + (alpha_0-1.)*np.sum(E_pi)
		eq74 = 0.5 * np.sum( ndim*np.log(beta_0/(2.*np.pi)) + E_lam - ndim*beta_0/beta_k - beta_0 * nu_k * pdot((m_k-m_0)[:,None,:],pdot(W_k,(m_k-m_0)[:,:,None])))
		eq74 += nstates * stats.wishart_ln_B(W_0,np.array((nu_0))) + (nu_0 - ndim -1.)/2. * np.sum(E_lam) - 0.5 * np.sum(nu_k * np.trace(pdot(Winv_0[None,...],W_k),axis1=-2,axis2=-1))
		eq75 = np.sum(np.sum(r_nk*np.log(r_nk+1e-300),axis=1),axis=0)
		eq76 = np.sum((alpha_k-1.)*E_pi) + special.gammaln(alpha_k.sum()) - np.sum(special.gammaln(alpha_k))
		eq77 = 0.5 *np.sum(E_lam + ndim*np.log(beta_k/(2.*np.pi)) - ndim - 2.*stats.wishart_entropy(W_k,nu_k))
		lowerbound =  eq71+eq72+eq73+eq74-eq75-eq76-eq77
		return lowerbound#,[eq71,eq72,eq73,eq74,eq75,eq76,eq77]
	def kmeans(x,nstates,nrestarts=5):
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
		return [pi_k[xsort],r_nk[:,xsort],mu_k[xsort],sig_k[xsort]]
	
	#### Initialize
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

	it = 0
	while 1:
		if it > maxiter or finished_counter > 3:
			break
		
		N_k = np.sum(r_nk,axis=0)
		xbar_k = np.sum(r_nk[:,:,None]*x[:,None,:],axis=0)/(N_k+1e-16)[:,None]
		S_k = np.sum(r_nk[:,:,None,None]*pdot((x[:,None,:] - xbar_k[None,:,:])[:,:,:,None],(x[:,None,:] - xbar_k[None,:,:])[:,:,None,:]),axis=0)/(N_k+1e-16)[:,None,None]

		nu_k = nu_0+ N_k
		beta_k = beta_0 + N_k
		alpha_k = alpha_0 + N_k
		m_k = (beta_0*m_0 + N_k[:,None]*xbar_k)/beta_k[:,None]
		W_k = np.linalg.inv( Winv_0 + N_k[:,None,None]*S_k + (beta_0*N_k/(beta_0 + N_k))[:,None,None] * pdot((xbar_k - m_0)[:,:,None],(xbar_k - m_0)[:,None,:]) )
		
		E_lam = np.sum(special.psi((nu_k[:,None] + 1. - np.linspace(1,ndim,ndim)[None,:])/2.),axis=1) + ndim*np.log(2.) + np.log(np.linalg.det(W_k))
		E_pi = special.psi(alpha_k) - special.psi(alpha_k.sum())
		E_mulam = ndim/beta_k[None,:] + nu_k[None,:] * pdot((x[:,None,None,:] - m_k[None,:,None,:]),pdot(W_k[None,:,:,:],(x[:,None,:,None] - m_k[None,:,:,None])))[:,:,0,0]

		rho_nk = E_pi[None,:] + .5 * E_lam[None,:] - ndim/2.*np.log(2.*np.pi) - .5*E_mulam
		rho_nk -= rho_nk.max(1)[:,None]
		rho_nk = np.exp(rho_nk)
		r_nk = rho_nk/np.sum(rho_nk,axis=1)[:,None]


		l = calc_lowerbound(nstates,ndim,ntraces,alpha_0,beta_0,nu_0,W_0,Winv_0,m_0,r_nk,N_k,xbar_k,S_k,m_k,W_k,alpha_k,beta_k,nu_k,E_lam,E_pi,E_mulam)
		
		if not np.ndim(lb):
			lb = np.array((it,l))[None,:]
		else:
			lb = np.append(lb,np.array((it,l))[None,:],axis=0)

		##A few threshold options: equivalent, rel. change, abs. change
		# if it > 1 and lb[it-1,1] == lb[it,1]:
		if it > 1 and (np.abs((lb[it,1]-lb[it-1,1])/lb[it,1]) < lowerbound_threshold):
		# if it > 1 and (np.abs((lb[it,1]-lb[it-1,1])) < lowerbound_threshold):
			finished_counter += 1
		else:
			finished_counter = 0
			
		it += 1
	
	xsort = alpha_k.argsort()[::-1]
	return [alpha_k[xsort],r_nk[:,xsort],m_k[xsort],beta_k[xsort],nu_k[xsort],W_k[xsort],S_k[xsort],lb,state_log]


if 1:
	states = 2
	dim = 2
	mu = np.random.uniform(-10,10,size=(states,dim))
	print mu
	
	covar = np.random.uniform(.01,.05,size=(states,dim))[...,None]*np.repeat(np.identity(dim)[None,:],states,axis=0)**2.
	nums = np.random.randint(200,300,size=states)
	f = nums/nums.astype('f').sum()
	xsort =f.argsort()[::-1]
	mu = mu[xsort]
	covar = covar[xsort]
	nums = nums[xsort]
	f = f[xsort]


	x = None
	for m,c,n in zip(mu,covar,nums):
		if x is None:
			x = stats.rv_multivariate_normal(m,c,n)
		else:
			x = np.append(x,stats.rv_multivariate_normal(m,c,n),axis=0)
	np.savetxt('x.dat',x)
	np.savetxt('f.dat',f)
else:
	x = np.loadtxt('x.dat')
	f = np.loadtxt('f.dat')
	states = 2
	dim = 2
print f


import matplotlib.pyplot as plt
f1,a1 = plt.subplots(1)
a1.scatter(mu[0,0],mu[0,1],color='r')
a1.scatter(x[:,0],x[:,1],alpha=.5)

# y = kmeans_new(x,4)
# xsort = y[0].argsort()[::-1]
# print "\n",y[0]
# print y[0]
# print y[1]
# for i in range(y[0].size):
	# plot_cov_ellipse(y[3][i],y[2][i])
def posterior_marginal_mu(x,n,mu,beta,nu,W):
	d = 1.
	mui = mu[n]
	Wi = W[n,n]/(beta*(nu-d+1.))
	nui = nu
	y1 = special.gamma((nui+d)/2.)*(np.pi*nui)**(-d/2.)
	y2 = np.abs(Wi)**-.5 / special.gamma(nui/2.)
	y3 = (1. + 1/nui*((x-mui)/Wi*(x-mui))) ** (-.5*(nui+d))
	return y1 * y2 * y3
	


def multivar_t_pdf(x,nu,sig):
	d = float(nu.size)
	y1 = special.gamma((nu+d)/2.)*(np.pi*nu)**(-d/2.)
	y2 = np.linalg.det(sig)**-.5 / special.gamma(nu/2.)
	y3 = (1. + 1/nu*pdot((x-mu)[...,None,:],pdot(np.linalg.inv(sig),(x-mu)[...,:,None]))[...,0]) ** (-.5*(nu+d))
	return y1*y2*y3
	

# import time
# t0 = time.time()
# print "\n"
# smax = 5
# rst = np.linspace(1,smax,smax)
# lb = np.zeros_like(rst)
# for i in rst:
# 	yy = []
# 	for j in range(5):
# 		y = variational_gmm_new(x,int(i),lbfxn=calc_lowerbound_new)
# 		if i == 1 and j == 0:
# 			ymaxmax = y
# 		yy.append(y)
#
# 	ymax = yy[0]
# 	for y in yy:
# 		if y[-2][-1][1] > ymax[-2][-1][1]:
# 			ymax = y
# 		if ymax[-2][-1][1] > ymaxmax[-2][-1][1]:
# 			ymaxmax = ymax
#
# 	if i == states:
# 		for j in range(states):
# 			plot_cov_ellipse(np.linalg.inv(ymax[4][j][...,None]*ymax[5][j]),ymax[2][j])
#
# 	lb[i - 1] = ymax[-2][-1][1]
# 	print i,ymax[-2][-1][1]
# print lb
# # print ymaxmax[-2][-1][1]
# print ymaxmax[0]/ymaxmax[0].sum()
# t1 = time.time()
# print t1-t0
#
# for i in range(ymaxmax[0].size):
# 	# plot_cov_ellipse(np.linalg.inv(ymaxmax[4][i][...,None]*ymaxmax[5][i]),ymaxmax[2][i])
# 	plt.scatter(ymaxmax[2][i][0],ymaxmax[2][i][1],color='r')
#
# plt.figure()
# [alpha_k[xsort],r_nk[:,xsort],m_k[xsort],beta_k[xsort],nu_k[xsort],W_k[xsort],S_k[xsort],lb,state_log]
f,ax = plt.subplots(2)
y = variational_gmm_new(x,2)
for j in range(y[0].size):
	for i,a in zip(range(2),ax):
		x = np.linspace(-10,10,200001)
		# m b nu w
		nu = y[4][j]
		sig = y[5][j][i,i]/(y[3][j]*(y[4][j]-1.+1.))
		print (nu/(nu-2.))
		a.semilogy(x,posterior_marginal_mu(x,i,y[2][j],y[3][j],y[4][j],y[5][j]))
		a.axvline(x=mu[j,i],color='r')
plt.show()
