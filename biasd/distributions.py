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
	
class parameter_collection(object):
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
		self.e1 = e1
		self.e2 = e2
		self.sigma = sigma
		self.k1 = k1
		self.k2 = k2
		self.labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\sigma$', r'$k_1$', r'$k_2$']
		
		self.okay = self.check_dists()
	
	def check_dists(self):
		self.okay = True
		for d,dname in zip([self.e1,self.e2,self.sigma,self.k1,self.k2], ('e1','e2','sigma','k1','k2')):
			if isinstance(d,_distribution):
				if not d.okay:
					self.okay = False
					raise ValueError('The prior for '+dname+' is malformed. Check the parameters values.')
			else:
				self.okay = False
				raise ValueError('The prior for '+dname+' is not a _distribution.')
		
	def rvs(self,n=1):
		# if self.okay and isinstance(n,int):
			rout = np.zeros((5,n))
			rout[0] = self.e1.rvs(n)
			rout[1] = self.e2.rvs(n)
			rout[2] = self.sigma.rvs(n)
			rout[3] = self.k1.rvs(n)
			rout[4] = self.k2.rvs(n)
			return rout
	
	def lnpdf(self,theta):
		e1,e2,sigma,k1,k2 = theta
		# if self.okay:
		return self.e1.lnpdf(e1) + self.e2.lnpdf(e2) + self.sigma.lnpdf(sigma) + self.k1.lnpdf(k1) + self.k2.lnpdf(k2)
		
	def mean(self):
		return np.array((self.e1.mean(),self.e2.mean(),self.sigma.mean(),self.k1.mean(),self.k2.mean()))
		
	def mode(self):
		return np.array((self.e1.mode(),self.e2.mode(),self.sigma.mode(),self.k1.mode(),self.k2.mode()))
	def variance(self):
		return np.array((self.e1.variance(),self.e2.variance(),self.sigma.variance(),self.k1.variance(),self.k2.variance()))
		
	def format(self):
		names = [d.name for d in [self.e1,self.e2,self.sigma,self.k1,self.k2]]
		params = [d.parameters.tolist() for d in [self.e1,self.e2,self.sigma,self.k1,self.k2]]
		return names,params
	#
	# @staticmethod
	# def new_from_smd(names,parameters):
	# 	dist_dict = {'beta':beta, 'dirichlet':dirichlet, 'empty':empty,'gamma':gamma, 'normal':normal, 'uniform':uniform}
	# 	p = [dist_dict[name](*params) for name,params in zip(names,parameters)]
	# 	return parameter_collection(*p)
		
class empty_parameter_collection(parameter_collection):
	'''A fake collection of distribution functions for facilitating use of BIASD without any priors.'''
	def __init__(self):
		super(empty_parameter_collection,self).__init__(*[empty() for _ in range(5)])
		self.okay = True
		
	# Overload to separate e1 and e2 s.t. e1 < e2....
	def rvs(self,n=1):
		# if self.okay and isinstance(n,int):
			rout = np.zeros((5,n))
			rout[0] = self.e1.rvs(n) - 1.
			rout[1] = self.e2.rvs(n) + 1.
			rout[2] = self.sigma.rvs(n)
			rout[3] = self.k1.rvs(n)
			rout[4] = self.k2.rvs(n)
			return rout
	def lnpdf(self,theta):
		if theta.ndim == 1:
			return 0.
		else:
			return np.zeros((theta.shape[1]))

class viewer(object):
	"""
	Allows you to view BIASD parameter probability distributions
	
	Input:
		* parameter_collection is a biasd.distributions.parameter_collection

	Example:
	
	.. code-block:: python
	
		from biasd import distributions as bd
		
		# Make the parameter_collection
		e1 = bd.normal(5.,1.)
		e2 = bd.beta(95.,5.)
		sigma = bd.gamma(5.,100.)
		k1 = bd.gamma(1.,1.)
		k2 = bd.uniform(-1,1.)
		d = bd.parameter_collection(e1,e2,sigma,k1,k2)
	
		# Start the viewer
		v = bd.viewer(d)
	"""
	import matplotlib.pyplot as plt
	
	class mpl_button(object):
		from matplotlib.widgets import Button
		
		def __init__(self,identity,name,x_loc,y_loc,viewer):
			self.height = .1
			self.width = .2
			self.ax = viewer.plt.axes([x_loc,y_loc,self.width,self.height])
			self.identity = identity
			self.name = name
			self.button = self.Button(self.ax,name)
			self.viewer = viewer
			self.button.on_clicked(self.clicked)
		def clicked(self,event):
			if self.identity != self.viewer.selected:
				self.viewer.deselect()
				self.viewer.select(self.identity)

	def __init__(self,data):
		if not isinstance(data,parameter_collection):
			raise ValueError('This is not a valid parameter collection')
		else:
			self.data = data
			self.f = viewer.plt.figure("Parameter Probability Distribution Function Viewer", 				figsize=(8,6))
			
			self.e1 = self.mpl_button("e1",r"$\varepsilon_1$",0.0,0.9,self)
			self.e2 = self.mpl_button("e2",r"$\varepsilon_2$",0.2,0.9,self)
			self.sigma = self.mpl_button("sigma",r"$\sigma$",0.4,0.9,self)
			self.k1 = self.mpl_button("k1",r"$k_1$",0.6,0.9,self)
			self.k2 = self.mpl_button("k2",r"$k_2$",0.8,0.9,self)
			
			self.colors = dict(list(zip(['e1','e2','sigma','k1','k2'], 				['purple','yellow','green','cyan','orange'])))
			self.xlabels = dict(list(zip(['e1','e2','sigma','k1','k2'], 				['Signal','Signal','Signal Noise',r'Rate Constant (s$^{-1}$)',r'Rate Constant (s$^{-1}$)'])))
			
			self.ax = viewer.plt.axes([.1,.1,.8,.7])
			self.ax.set_yticks((),())
			self.ax.set_ylabel('Probability',fontsize=18,labelpad=15)
			self.line = self.ax.plot((0,0),(0,0),color=self.colors['e1'],lw=1.5)[0]
			self.fill = self.ax.fill_between((0,0),(0,0),color='white')
			self.select("e1")
			viewer.plt.show()
			
	def deselect(self):
		self.__dict__[self.selected].ax.set_axis_bgcolor('lightgrey')
		viewer.plt.draw()
		self.selected = None

	def select(self,identity):
		self.__dict__[identity].ax.set_axis_bgcolor('lightblue')
		self.selected = identity
		self.update_plot()
		
	def update_plot(self):
		dist = self.data.__dict__[self.selected]
		distx = dist.get_ranged_x(1001)
		disty = dist.pdf(distx)
		
		self.line.set_xdata(distx)
		self.line.set_ydata(disty)
		self.line.set_color('k')
		# self.line.set_color(self.colors[self.selected])
		
		for collection in (self.ax.collections):
			self.ax.collections.remove(collection)
		self.ax.fill_between(distx,disty,color=self.colors[self.selected], 			alpha=0.75)
		
		if isinstance(dist,beta):
			self.ax.set_xlim(0,1)
		elif isinstance(dist,gamma):
			self.ax.set_xlim(0,distx[-1])
		else:
			self.ax.set_xlim(distx[0],distx[-1])
		self.ax.set_ylim(0.,disty.max()*1.2)
		self.ax.set_xlabel(self.xlabels[self.selected],fontsize=18)
		self.ax.set_title(dist.label_parameters[0]+": "+str(dist.parameters[0])+", "+dist.label_parameters[1]+": "+str(dist.parameters[1])+r", $E[x] = $"+str(dist.mean()))
		self.f.canvas.draw_idle()
		
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
	return parameter_collection(e1,e2,sigma,k1,k2)

#### Guess Priors

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
	return parameter_collection(e1,e2,sigma,k1,k2)


