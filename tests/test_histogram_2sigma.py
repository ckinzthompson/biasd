import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_histogram_2sigma():
	b.likelihood.use_python_numba_ll_2sigma()

	data = ssm.testdata(nmol=3,nt=500).flatten()
	tau = .1
	print(f"No. datapoints {data.size}")

	guess = np.array((0.,1.,.08,.08,1.,2.))
	theta,covars = b.histogram.fit_histogram(data,tau,guess)
	fig,ax = b.plot.likelihood_hist(data,tau,theta)
	fig.savefig(os.path.join(fdir,'fig_histogram_2sigma.png'))
	plt.close()

	## Check stats
	truth = np.array((0.,1.,.05,.05,3.,8.))
	delta = np.abs(theta-truth)
	target = np.array((.1,.1,.02,.02,2.,2.))
	assert np.all(delta < target)

if __name__ == '__main__':
	test_histogram_2sigma()