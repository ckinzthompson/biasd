import numpy as np
import matplotlib.pyplot as plt
import corner

from .likelihood import nosum_log_likelihood
from .mcmc import get_samples

def trace_likelihood(theta,tau,nsamples=1000,range=None):
	if range is None:
		xmin = theta[0] - 3*theta[2]
		xmax = theta[1] + 3*theta[2]
	else:
		xmin = range[0]
		xmax = range[1]

	x = np.linspace(xmin,xmax,nsamples)
	y = np.exp(nosum_log_likelihood(theta,x,tau))
	return x,y

def _sample_hist(data,tau,theta,samples=None,nbins=101,range=None,nsamples=100):
	plt.rc('font', family='Arial', size=8)
	plt.rc('axes', linewidth=1.)
	plt.rc('lines', linewidth=1.)

	if range is None:
		xmin = theta[0] - 3*theta[2]
		xmax = theta[1] + 3*theta[2]
	else:
		xmin = range[0]
		xmax = range[1]

	if not samples is None:
		q = []
		for stheta in samples:
			x,y = trace_likelihood(stheta,tau)
			q.append(y)
		l,m,h = np.percentile(np.array(q),[2.5,50,97.5],axis=0)
	else:
		x,m = trace_likelihood(theta,tau)

	fig,ax = plt.subplots(1)
	ax.hist(data,bins=nbins,range=(xmin,xmax),density=True,alpha=.8,histtype='step',color='tab:blue')
	if not samples is None:
		ax.fill_between(x,l,h,color='tab:orange',alpha=.6)
	ax.plot(x,m,color='tab:orange',alpha=.8)

	ax.set_ylabel('Probability Density',fontsize=10)
	ax.set_xlabel('Signal',fontsize=10)

	fig.set_figheight(1.8)
	fig.set_figwidth(2.2)
	fig.tight_layout()
	return fig,ax

def likelihood_hist(data,tau,theta,nbins=101,range=None,nsamples=100):
	samples = None
	fig,ax = _sample_hist(data,tau,theta,samples,nbins,range,nsamples)
	return fig,ax

def laplace_hist(data,tau,posterior,nbins=101,range=None,nsamples=100):
	theta = posterior.mu
	samples = posterior.samples(nsamples)
	fig,ax = _sample_hist(data,tau,theta,samples,nbins,range,nsamples)
	return fig,ax

def mcmc_hist(data,tau,sampler,nbins=101,range=None,nsamples=100):
	samples = get_samples(sampler,uncorrelated=True)
	theta = samples.mean(0)
	fig,ax = _sample_hist(data,tau,theta,samples,nbins,range,nsamples)
	return fig,ax

def mcmc_corner(sampler):
	"""
	Use the python package called corner <https://github.com/dfm/corner.py> to make some very nice corner plots (joints and marginalized) of posterior in the 5-dimensions used by the two-state BIASD posterior.

	Input:
		* `samples` is a (N,5) `np.ndarray`
	Returns:
		* `fig` which is the handle to the figure containing the corner plot
	"""
	samples = get_samples(sampler,uncorrelated=True,verbose=False)
	samples[:,2:] = np.log10(samples[:,2:])
	labels = [r'$\epsilon_1$', r'$\epsilon_2$', r'$\log_{10}\left(\sigma\right)$', r'$\log_{10}\left(k_1\right)$', r'$\log_{10}\left(k_2\right)$']
	fig = corner.corner(samples, labels=labels,quantiles=[.025,.50,.975],show_titles=True,)#,levels=(1-np.exp(-0.5),))
	return fig


'''
import biasd as b

theta,covars = b.likelihood.fit_histogram(data,tau)
fig,ax = plot_likelihood_hist(data,tau,theta)
plt.show()

e1 = b.distributions.normal(0.,.2)
e2 = b.distributions.normal(1.,.2)
sigma = b.distributions.loguniform(.01,.5)
k12 = b.distributions.loguniform(1e-1*tau,1e3*tau)
k21 = b.distributions.loguniform(1e-1*tau,1e3*tau)
prior = b.distributions.parameter_collection(e1,e2,sigma,k12,k21)
posterior = b.laplace.laplace_approximation(data,prior,tau,guess=theta,verbose=True,ensure=True)
fig,ax = plot_laplace_hist(data,tau,posterior)
plt.show()

'''