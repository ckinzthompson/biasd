"""
.. module:: distributions

	:synopsis: Contains probability distributions for use with BIASD prior and posterior distributions


"""

# import matplotlib.pyplot as plt
import numpy as np
np.seterr(all="ignore")
from scipy import special as special

class _distribution(object):
	"""
	Parameters should already have been checked.
	PDF/PMF should check support
	"""
	
	__minimum__ = 0.
	__small__ = 1e-4
	
	def __init__(self,parameters,label_parameters,lnpdf,mean,variance,rvs,okay):
		self._lnpdf_fxn = lnpdf
		self.parameters = parameters
		self.label_parameter = label_parameters
		self.support_parameters = self.support_parameters
		self.support = support
		self._mean = mean
		self._variance = variance
		# self._moment2param_fxn = moment2param
		self._rvs_fxn = rvs
		self.okay = okay

	def check_params(self):
		self.okay = True
		for i,r in zip(self.parameters,self.support_parameters):
			if i <= r[0] or i >= r[1]:
				self.okay = False
				
	def pdf(self,x):
		"""
		Returns the probability distribution/mass function
		"""
		if self.okay:
			return np.exp(self._lnpdf_fxn(x,self.parameters,self.support))
	def lnpdf(self,x):
		"""
		Returns the natural log of the probability distribution/mass function
		"""
		if self.okay:
			return self._lnpdf_fxn(x,self.parameters,self.support)
	def rvs(self,size):
		"""
		Returns random variates in the shape of size (tuple)
		"""
		if self.okay:
			return self._rvs_fxn(size,self.parameters)
			
	def mean(self):
		"""
		First Moment
		"""
		if self.okay:
			return self._mean(self.parameters)
	def variance(self):
		"""
		Second moment - square of first moment:
		E[x^2] - E[x]^2
		"""
		if self.okay:
			return self._variance(self.parameters)
	def mode(self):
		"""
		Mode
		"""
		if self.okay:
			return self._mode(self.parameters)
	
	def get_xlim(self):
		if self.okay:
			return self._get_xlim(self.parameters)
	
	def get_ranged_x(self,n):
		"""
		Returns an array of n datapoints that covers most of the PDF mass
		"""
		if self.okay:
			return self._get_ranged_x(self.parameters,n)

class beta(_distribution):
	"""
	The beta distribution is often used for probabilities or fractions.
	
	It is :math:`p(x\\vert\\alpha ,\\beta) = \\frac{ x^{\\alpha-1}(1-x)^{\\beta-1}}{B(\\alpha ,\\beta)}`
	"""
	def __init__(self,alpha,beta):
		self.name = 'beta'
		# initial loading/defining parameters
		self.parameters = np.array((alpha,beta),dtype='f')
		self.support_parameters = np.array(((_distribution.__minimum__, np.inf), (_distribution.__minimum__, np.inf)))
		self.support = np.array((_distribution.__minimum__, 1.-_distribution.__minimum__))

		self.label_parameters = [r"$\alpha$",r"$\beta$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return beta(*parameters)
		
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		return parameters[0]/(parameters[0]+parameters[1])
	
	@staticmethod
	def _variance(parameters):
		a,b = parameters
		return a*b/((a*b)**2.*(a+b+1.))
	
	@staticmethod
	def _mode(parameters):
		a,b = parameters
		if a > 1. and b > 1.:
			return (a-1.)/(a+b-2.)
		else:
			return np.nan
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		a,b = parameters
		if isinstance(x,np.ndarray):
			keep = (x >= support[0])*(x<=support[1])
			y = (a-1.)*np.log(x)+(b-1.)*np.log(1.-x) - special.betaln(a,b)
			y[np.nonzero(~keep)] = -np.inf
		else:
			keep = (x >= support[0])*(x<=support[1])
			if keep:
				y = (a-1.)*np.log(x)+(b-1.)*np.log(1.-x) - special.betaln(a,b)
			else:
				y = -np.inf
		return y
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		a,b = parameters
		return np.random.beta(a,b,size=size)
		
	@staticmethod
	def _moment2param_fxn(first,second):
		variance = second - first**2.
		alpha = first*(first*(1.-first)/variance-1.)
		beta = (1.-first)*(first*(1.-first)/variance-1.)
		return np.array([alpha,beta])
	
	@staticmethod
	def _get_xlim(parameters):
		l = special.betaincinv(parameters[0],parameters[1],_distribution.__small__)
		h = special.betaincinv(parameters[0],parameters[1],1.-_distribution.__small__)
		return np.array((l,h))
		
	@staticmethod
	def _get_ranged_x(parameters,n):
		l,h = beta._get_xlim(parameters)
		return np.linspace(l,h,int(n))
		
class dirichlet(_distribution):
	"""
	The dirichlet distribution is often used for probabilities or fractions.
	
	It is :math:`p(x\\vert \\vec {\\alpha}) = \\frac{1}{B(\\vec{\\alpha})}\\prod\\limits_{i=1}^k x_i ^{\\alpha_i -1}`
	"""
	def __init__(self,alpha):
		self.name = 'dirichlet'
		# initial loading/defining parameters
		self.parameters = np.array(alpha,dtype='f')
		self.support_parameters = np.array(((_distribution.__minimum__, np.inf), (_distribution.__minimum__, np.inf)))
		self.support = np.array((_distribution.__minimum__, 1.-_distribution.__minimum__))

		self.label_parameters = [r"$\vec{\alpha}$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return dirichlet(*parameters)
		
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		return parameters/parameters.sum()
	
	@staticmethod
	def _variance(parameters):
		a = parameters
		a0 = parameters.sum()
		return a *(a0-a)/(a0*a0*(a0+1.))
	
	@staticmethod
	def _mode(parameters):
		a = parameters
		a0 = a.sum()
		if np.all(a > 1):
			return (a-1.)/(a0-a.size)
		else:
			return np.nan
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		a = parameters
		if isinstance(x,np.ndarray):
			if np.all(a > 0) and np.all(x >= 0.) and np.all(x <= 1.) and x.sum() == 1.:
				y = - (np.sum(special.gammaln(a)) - special.gammaln(a.sum()))
				y += np.sum((a-1.)*np.log(x))
				if np.isfinite(y):
					return y
		return -np.inf
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		a = parameters
		return np.random.dirichlet(a,size=size)
		
	# @staticmethod
	# def _moment2param_fxn(first,second):
	# 	variance = second - first**2.
	# 	alpha = first*(first*(1.-first)/variance-1.)
	# 	beta = (1.-first)*(first*(1.-first)/variance-1.)
	# 	return np.array([alpha,beta])
	
	@staticmethod
	def _get_xlim(parameters):
		l = 0.#special.betaincinv(parameters[0],parameters[1],_distribution.__small__)
		h = 1.#special.betaincinv(parameters[0],parameters[1],1.-_distribution.__small__)
		return np.array((l,h))
		
	@staticmethod
	def _get_ranged_x(parameters,n):
		l,h = 0.,1.#beta._get_xlim(parameters)
		return np.linspace(l,h,int(n))

class empty(_distribution):
	def __init__(self,a=0,b=0):
		self.name = 'empty'
		self.parameters = np.array([a,b])
		self.support_parameters = np.array([[None,None],[None,None]])
		self.support = np.array([None,None])
		self.label_parameters = ["None","None"]
		self.okay = True
	
	def check_params(self):
		self.okay = True
		
	@staticmethod
	def new(parameters):
		return empty()
	@staticmethod
	def _mean(parameters):
		return 0.
	@staticmethod
	def _variance(parameters):
		return 0.
	@staticmethod
	def _mode(parameters):
		return 0.
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		return x*-np.inf
	@staticmethod
	def _rvs_fxn(size,parameters):
		return np.random.rand(size)
	@staticmethod
	def _moment2param_fxn(first,second):
		return np.array([0.,0.])
	@staticmethod
	def _get_xlim(parameters):
		return np.array((0.,1.))
	@staticmethod
	def _get_ranged_x(parameters,n):
		return np.linspace(0.,1.,int(n))
		
class gamma(_distribution):
	"""
	The gamma distribution is often used for compounded times.
	
	It is :math:`p(x\\vert\\alpha ,\\beta) = \\frac{ \\beta^\\alpha x^{\\alpha - 1} e^{-\\beta x} }{\\Gamma(\\alpha)}`

	Parameters are alpha (shape), and beta (rate)
	"""
	def __init__(self,alpha,beta):
		self.name = 'gamma'
		# initial loading/defining parameters
		self.parameters = np.array((alpha,beta),dtype='f')
		self.support_parameters = np.array(((_distribution.__minimum__,np.inf), (_distribution.__minimum__,np.inf)))
		self.support = np.array((_distribution.__minimum__, np.inf))

		self.label_parameters = [r"$\alpha$",r"$\beta$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return gamma(*parameters)
	
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		return parameters[0]/parameters[1]
	
	@staticmethod
	def _variance(parameters):
		return parameters[0]/(parameters[1]**2.)
	
	@staticmethod
	def _mode(parameters):
		a,b = parameters
		if a >= 1.:
			return (a-1.)/b
		else:
			return np.nan
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		a,b = parameters
		if isinstance(x,np.ndarray):
			keep = (x >= support[0])*(x<=support[1])
			y = a*np.log(b)+(a-1.)*np.log(x)-b*x-special.gammaln(a)
			y[np.nonzero(~keep)] = -np.inf
		else:
			keep = (x >= support[0])*(x<=support[1])
			if keep:
				y = a*np.log(b)+(a-1.)*np.log(x)-b*x-special.gammaln(a)
			else:
				y = -np.inf
		return y
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		a,b = parameters
		return np.random.gamma(shape=a,scale=1./b,size=size)
		
	@staticmethod
	def _moment2param_fxn(first,second):
		variance = second - first**2.
		alpha = first*first/variance
		beta = first/variance
		return np.array([alpha,beta])

	@staticmethod
	def _get_xlim(parameters):
		l = special.gammaincinv(parameters[0],_distribution.__small__)/parameters[1]
		h = special.gammaincinv(parameters[0],1.-_distribution.__small__)/parameters[1]
		return np.array((l,h))

	@staticmethod
	def _get_ranged_x(parameters,n):
		l,h = gamma._get_xlim(parameters)
		return np.linspace(l,h,int(n))

class normal(_distribution):
	"""
	The normal/Gaussian distribution is useful for everything
	Parameters are mean, and the standard deviation
	
	It is :math:`p(x\\vert\\mu ,\\sigma) = \\frac{ 1}{\\sqrt{2\\pi\\sigma^2}}e^{\\frac{-1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}`
	"""
	def __init__(self,mu,sigma):
		self.name = 'normal'
		# initial loading/defining parameters
		self.parameters = np.array((mu,sigma),dtype='f')
		self.support_parameters = np.array(((-np.inf,np.inf), (_distribution.__minimum__,np.inf)))
		self.support = np.array((-np.inf, np.inf))

		self.label_parameters = [r"$\mu$",r"$\sigma$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return normal(*parameters)
	
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		return parameters[0]
	
	@staticmethod
	def _variance(parameters):
		return parameters[1]**2.
	
	@staticmethod
	def _mode(parameters):
		return parameters[0]
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		mu,sigma = parameters
		if isinstance(x,np.ndarray):
			keep = (x >= support[0])*(x<=support[1])
			y = -.5*np.log(2.*np.pi)-np.log(sigma) - .5 * ((x-mu)/sigma)**2.
			y[np.nonzero(~keep)] = -np.inf
		else:
			keep = (x >= support[0])*(x<=support[1])
			if keep:
				y = -.5*np.log(2.*np.pi)-np.log(sigma) - .5 * ((x-mu)/sigma)**2.
			else:
				y = -np.inf
		return y
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		mu,sigma = parameters
		return np.random.normal(loc=mu,scale=sigma,size=size)
		
	@staticmethod
	def _moment2param_fxn(first,second):
		variance = second - first**2.
		mu = first
		sigma = np.sqrt(variance)
		return np.array([mu,sigma])
	
	@staticmethod
	def _get_xlim(parameters):
		l = parameters[0] + parameters[1]*np.sqrt(2.)*special.erfinv(2.*(_distribution.__small__)-1.)
		h = parameters[0] + parameters[1]*np.sqrt(2.)*special.erfinv(2.*(1.-_distribution.__small__)-1.)
		return np.array((l,h))
	
	@staticmethod
	def _get_ranged_x(parameters,n):
		l,h = normal._get_xlim(parameters)
		return np.linspace(l,h,int(n))

class uniform(_distribution):
	"""
	The uniform distribution is useful limiting ranges
	Parameters are a (lower bound), and b (upper bound)
	
	It is :math:`p(x\\vert a , b) = \\frac{1}{b-a}`
	"""
	def __init__(self,a,b):
		self.name = 'uniform'
		# initial loading/defining parameters
		self.parameters = np.array((a,b),dtype='f')
		self.support_parameters = np.array(((-np.inf,b), (a,np.inf)))
		self.support = np.array((a,b))

		self.label_parameters = [r"$a$",r"$b$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return uniform(*parameters)
		
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		return .5 *(parameters[0]+parameters[1])
	
	@staticmethod
	def _variance(parameters):
		a,b = parameters
		return 1./12. *(b-a)**2.
	
	@staticmethod
	def _mode(parameters):
		# not really correct, but w/e... you gotta pick something
		return uniform._mean(parameters)
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		a,b = parameters
		if isinstance(x,np.ndarray):
			keep = (x >= support[0])*(x<=support[1])
			y = -np.log(b-a)*keep
			if np.any(~keep):
				y[np.nonzero(~keep)] = -np.inf
		else:
			keep = (x >= support[0])*(x<=support[1])
			if keep:
				y = -np.log(b-a)
			else:
				y = -np.inf
		return y
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		a,b = parameters
		return np.random.uniform(a,b+_distribution.__minimum__,size=size)
		
	@staticmethod
	def _moment2param_fxn(first,second):
		variance = second - first**2.
		a = first - np.sqrt(3.*variance)
		b = first + np.sqrt(3.*variance)
		return np.array([a,b])
	
	@staticmethod
	def _get_xlim(parameters):
		return parameters
	
	@staticmethod
	def _get_ranged_x(parameters,n):
		return np.linspace(parameters[0],parameters[1],int(n))
	
class loguniform(_distribution):
	"""
	The uniform distribution is useful limiting ranges
	Parameters are a (lower bound), and b (upper bound)
	
	It is :math:`p(x\\vert a , b) = \\frac{x^{-1}}{\ln b - \ln a}`
	"""
	def __init__(self,a,b):
		self.name = 'log uniform'
		# initial loading/defining parameters
		self.parameters = np.array((a,b),dtype='f')
		self.support_parameters = np.array(((-np.inf,b), (a,np.inf)))
		self.support = np.array((a,b))

		self.label_parameters = [r"$a$",r"$b$"]
		self.check_params()
	
	@staticmethod
	def new(parameters):
		return uniform(*parameters)
		
	# normal-specific things
	@staticmethod
	def _mean(parameters):
		a,b = parameters
		return (b-a)/(np.log(b)-np.log(a))
	
	@staticmethod
	def _variance(parameters):
		a,b = parameters
		return (b**2.-a**2.)/(2*(np.log(b)-np.log(a))) - ((b-a)/(np.log(b)-np.log(a)))**2.
	
	@staticmethod
	def _mode(parameters):
		# not really correct, but w/e... you gotta pick something
		a,b = parameters
		return np.sqrt(a*b) ## median
	
	@staticmethod
	def _lnpdf_fxn(x,parameters,support):
		a,b = parameters
		if isinstance(x,np.ndarray):
			keep = (x >= support[0])*(x<=support[1])
			y = np.zeros(x.size) - np.log(np.log(b)-np.log(a))
			y[keep] = -np.log(x[keep]) 
			y[np.bitwise_not(keep)] = -np.inf
		else:
			keep = (x >= support[0])*(x<=support[1])
			if keep:
				y = -np.log(x)-np.log(np.log(b)-np.log(a))
			else:
				y = -np.inf
		return y
	
	@staticmethod
	def _rvs_fxn(size,parameters):
		a,b = parameters
		return np.exp(np.random.uniform(np.log(a),np.log(b),size=size))
		
	@staticmethod
	def _moment2param_fxn(first,second):
		raise Exception('Not implemented')
	
	@staticmethod
	def _get_xlim(parameters):
		return parameters
	
	@staticmethod
	def _get_ranged_x(parameters,n):
		return np.linspace(parameters[0],parameters[1],int(n))


def convert_distribution(this,to_this_type_string):
	""" 
	Converts `this` distribution to `to_this_type_string` distribution
	
	Input:
		* `this` is a `biasd.distribution._distribution`
		* `to_this_type_string` is a string of 'beta', 'gamma', 'normal', or 'uniform'

	Returns:
		* a `biasd.distribution._distribution`
	
	Example:
	
	.. code-block:: python
	
		from biasd import distributions as bd
		import numpy as np
		import matplotlib.pyplot as plt
	
		n = bd.normal(.5,.01)
		x = np.linspace(0,1,10000)
		c = bd.convert_distribution(n,'gamma')
		cc = bd.convert_distribution(n,'uniform')
		ccc = bd.convert_distribution(n,'beta')
		plt.plot(x,n.pdf(x))
		plt.plot(x,c.pdf(x))
		plt.plot(x,cc.pdf(x))
		plt.plot(x,ccc.pdf(x))
		plt.yscale('log')
		plt.show()
	"""
	
	to_this_type = dict(list(zip(('beta','gamma','normal','uniform'),(beta,gamma,normal,uniform))))[to_this_type_string]
	
	params = to_this_type._moment2param_fxn(this.mean(), this.variance()+this.mean()**2.)
	
	return to_this_type(*params)

class collection(object):
	def __init__(self, **kwargs):
		self.parameters = {}
		for key, value in kwargs.items():
			setattr(self, key, value)
			self.parameters[key] = value
		self.labels = list(self.parameters.keys())
		self.num = len(self.labels)
		self.okay = self.check_dists()

	def check_dists(self):
		self.okay = True
		for i in range(self.num):
			label = self.labels[i]
			pd = self.parameters[label]
			if isinstance(pd,_distribution):
				if not pd.okay:
					self.okay = False
					raise ValueError(f'The prior for {label} is malformed. Check the parameter values.')
			else:
				self.okay = False
				raise ValueError(f'The prior for {label} is not a _distribution.')

	def rvs(self,n=1):
		if self.okay:
			rout = np.zeros((self.num,n))
			for i in range(self.num):
				label = self.labels[i]
				pd = self.parameters[label]
				rout[i] = pd.rvs(n)
			return rout
		else:
			print('Distributions are not OK!')
			for label in self.labels:
				print(label,self.parameters[label].okay)
			raise Exception('Failed')

	
	def lnpdf(self,theta):
		lnp = 0
		for i in range(self.num):
			label = self.labels[i]
			lnp += self.parameters[label].lnpdf(theta[i])
		return lnp
		
	def mean(self):
		return np.array([self.parameters[label].mean() for label in self.labels])
		
	def mode(self):
		return np.array([self.parameters[label].mode() for label in self.labels])
	
	def variance(self):
		return np.array([self.parameters[label].variance() for label in self.labels])
		
	def format(self):
		names = [self.parameters[label].name for label in self.labels]
		params = [self.parameters[label].parameters.tolist() for label in self.labels]
		return names,params

class collection_standard_1sigma(collection):
	"""
	A collection of distribution functions that are used for the BIASD two-state model as parameters for Bayesian inference. parameter_collection's are used as priors for BIASD.
	
	Input:
		* e1 is the probability distribution for :math:`\\epsilon_1`
		* e2 is the probability distribution for :math:`\\epsilon_2`
		* sigma is the probability distribution for :math:`\\sigma`
		* k1 is the probability distribution for :math:`k_1`
		* k2 is the probability distribution for :math:`k_2`
	"""
	
	def __init__(self,e1,e2,sigma,k1,k2):
		super().__init__(e1=e1, e2=e2, sigma=sigma, k1=k1, k2=k2)
		self.fancy_labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma$', r'$k_1$', r'$k_2$']
		self.check_dists()

class collection_standard_2sigma(collection):
	"""
	A collection of distribution functions that are used for the BIASD two-state model as parameters for Bayesian inference. parameter_collection's are used as priors for BIASD.
	
	Input:
		* e1 is the probability distribution for :math:`\\epsilon_1`
		* e2 is the probability distribution for :math:`\\epsilon_2`
		* sigma1 is the probability distribution for :math:`\\sigma_1`
		* sigma2 is the probability distribution for :math:`\\sigma_2`
		* k1 is the probability distribution for :math:`k_1`
		* k2 is the probability distribution for :math:`k_2`
	"""
	
	def __init__(self,e1,e2,sigma1,sigma2,k1,k2):
		super().__init__(e1=e1, e2=e2, sigma1=sigma1, sigma2=sigma2, k1=k1, k2=k2)
		self.fancy_labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma_1$', r'$\sigma_2$', r'$k_1$', r'$k_2$']
		self.check_dists()


## defaults
def uninformative_prior(data_range,timescale):
	"""
	Generate an uninformative prior probability distribution for BIASD.
	
	Input:
		* `data_range` is a list or array of the [lower,upper] bounds of the data
		* `timescale` is the frame rate (the prior will be centered here +/- 3 decades)
	
	Returns:
		* a flat `biasd.distributions.parameter_collection`
	"""
	lower,upper = data_range
	e1 = uniform(lower,(upper-lower)/2.+lower)
	e2 = uniform((upper-lower)/2.+lower,upper)
	sigma = gamma(1,1./((upper-lower)/10.))
	k1 = gamma(1.,timescale)
	k2 = gamma(1.,timescale)
	return collection_standard_1sigma(e1,e2,sigma,k1,k2)

def guess_prior(y,tau=1.):
	"""
	Generate a guess for the prior probability distribution for BIASD. This approach uses a Gaussian mixture model to learn both states and the noise, then it idealizes the trace, and calculates the transition probabilities. Rate constants are then calculated, and an attempt is made to correct these with virtual states.
	
	Input:
		* `y` is a `numpy.ndarray` of the time series
		* `tau` is the measurement period of the time series (i.e., inverse acquisition rate)
	
	Returns:
		* a guessed `biasd.distributions.parameter_collection`
	"""
	
	m1 = y.min()
	m2 = y.max()
	s = y.std()
	k12 = tau*1.
	k21 = tau*1.


	# Noise
	a = 3.
	b = 3./s

	a1 = 2.
	b1 = 2./k12
	a2 = 2.
	b2 = 2./k21

	e1 = normal(m1,s)
	e2 = normal(m2,s)
	sigma = gamma(a,b)
	k1 = gamma(a1,b1)
	k2 = gamma(a2,b2)
	return collection_standard_1sigma(e1,e2,sigma,k1,k2)


