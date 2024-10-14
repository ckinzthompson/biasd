import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_laplace_2sigma():
	b.likelihood.use_python_numba_ll_2sigma()


	data = ssm.testdata(nmol=3,nt=500).flatten()
	tau = .1
	print(f"No. datapoints {data.size}")

	e1 = b.distributions.normal(0.,.2)
	e2 = b.distributions.normal(1.,.2)
	sigma1 = b.distributions.loguniform(.01,.5)
	sigma2 = b.distributions.loguniform(.01,.5)
	k12 = b.distributions.loguniform(1e-1*tau,1e3*tau)
	k21 = b.distributions.loguniform(1e-1*tau,1e3*tau)
	prior = b.distributions.collection_standard_2sigma(e1,e2,sigma1,sigma2,k12,k21)

	guess = np.array((0.,1.,.08,.08,1.,2.))
	posterior = b.laplace.laplace_approximation(data,prior,tau,guess=guess,verbose=True,ensure=True)
	fig,ax = b.plot.laplace_hist(data,tau,posterior)
	fig.savefig(os.path.join(fdir,'fig_laplace_2sigma.png'))
	plt.close()

	## Check stats
	mu = posterior.mu
	truth = np.array((0.,1.,.05,.05,3.,8.))
	delta = np.abs(mu-truth)
	target = np.array((.1,.1,.02,.02,2.,2.))
	assert np.all(delta < target)

if __name__ == '__main__':
	test_laplace_2sigma()