import numpy as np
from scipy.integrate import quad
from scipy import special as special

def python_integrand(x,d,e1,e2,sigma1,sigma2,k1,k2,tau):
	"""
	Integrand for BIASD likelihood function
	"""
	#Ensures proper support
	if x < 0. or x > 1. or k1 <= 0. or k2 <= 0. or sigma1 <= 0. or sigma2 <= 0. or tau <= 0. or e1 >= e2:
		return 0.
	else:
		k = k1 + k2
		p1 = k2/k
		p2 = k1 /k
		y = 2.*k*tau * np.sqrt(p1*p2*x*(1.-x))
		z = p2*x + p1*(1.-x)
		varr = sigma1**2. * x + sigma2**2. *(1.-x)
		pf = 2.*k*tau*p1*p2*(special.i0(y)+k*tau*(1.-z)*special.i1(y)/y)*np.exp(-z*k*tau)
		py = 1./np.sqrt(2.*np.pi*varr)*np.exp(-.5/varr*(d-(e1*x+e2*(1.-x)))**2.) * pf
		return py

def python_integral(d,e1,e2,sigma1,sigma2,k1,k2,tau):
	"""
	Use Gaussian quadrature to integrate the BIASD integrand across df between f = 0 ... 1
	"""
	return quad(python_integrand, 0.,1.,args=(d,e1,e2,sigma1,sigma2,k1,k2,tau), limit=1000)[0]
python_integral = np.vectorize(python_integral)

def p_gauss(x,mu,sigma):
	return 1./np.sqrt(2.*np.pi*sigma**2.) * np.exp(-.5*((x-mu)/sigma)**2.)

def nosum_log_likelihood_scipy(theta,data,tau,device=None):
	"""
	Calculate the log of the BIASD likelihood function at theta using the data data given the time period of the data as tau.

	Python Version
	"""

	e1,e2,sigma,k1,k2 = theta
	p1 = k2/(k1+k2)
	p2 = 1.-p1
	out = python_integral(data,e1,e2,sigma,sigma,k1,k2,tau)
	peak1 = p_gauss(data,e1,sigma)
	peak2 = p_gauss(data,e2,sigma)
	out += p1*peak1*np.exp(-k1*tau)
	out += p2*peak2*np.exp(-k2*tau)

	#Don't use -infinity
	return np.log(out)

def log_likelihood_scipy(theta,data,tau,device=None):
	#print('py')
	return np.nansum(nosum_log_likelihood_scipy(theta,data,tau))