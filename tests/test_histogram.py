import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_histogram():
	b.likelihood.use_python_ll()

	data = ssm.testdata(nmol=3,nt=500).flatten()
	tau = .1
	print(f"No. datapoints {data.size}")

	e1 = b.distributions.normal(0.,.2)
	e2 = b.distributions.normal(1.,.2)
	sigma = b.distributions.loguniform(.01,.5)
	k12 = b.distributions.loguniform(1e-1*tau,1e3*tau)
	k21 = b.distributions.loguniform(1e-1*tau,1e3*tau)
	prior = b.distributions.parameter_collection(e1,e2,sigma,k12,k21)

	guess = np.array((0.,1.,.08,1.,2.))
	theta,covars = b.histogram.fit_histogram(data,tau,guess)
	fig,ax = b.plot.likelihood_hist(data,tau,theta)
	fig.savefig(os.path.join(fdir,'fig_histogram.png'))
	plt.close()

	## Check stats
	truth = np.array((0.,1.,.05,3.,8.))
	delta = np.abs(theta-truth)
	target = np.array((.1,.1,.02,2.,2.))
	assert np.all(delta < target)

if __name__ == '__main__':
	test_histogram()