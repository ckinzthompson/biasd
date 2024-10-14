import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import corner
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_constantepsilon():
	b.likelihood.use_python_numba_ll()

	#### Simulate data
	sigmas = np.array((.05,.05,.05,.05))
	k1s = np.array((1.,3.,10.,30.))
	k2s = np.array((30.,10.,3.,1.))

	datas = []
	for i in range(len(sigmas)):
		rates = np.zeros((2,2))
		rates[0,1] = k1s[i]
		rates[1,0] = k2s[i]
		emissions = np.array((0.,1.))
		noise = sigmas[i]
		nframes = 100
		dt = .1
		nmol = 50

		data = ssm.simulate_ensemble(rates,emissions,noise,nframes,dt,nmol)
		datas.append(data.flatten())

		# plt.hist(data.flatten(),range=(-.5,1.5),bins=201,histtype='step')
	# plt.show()


	#### Analyze

	## SETUP
	tau = dt

	pe1 = b.distributions.normal(0.,.01)
	pe2 = b.distributions.normal(1.,.01)
	pskk = []
	for i in range(len(datas)):
		psigma = b.distributions.loguniform(.04,.06)
		pk1 = b.distributions.loguniform(.01,100.)
		pk2 = b.distributions.loguniform(.01,100.)
		pskk.append([psigma,pk1,pk2])
	prior = b.constantepsilon.collection_constantepsilon(pe1,pe2,pskk)

	nwalkers = 8*(2+3*prior.ndim)
	sampler, initial_positions = b.constantepsilon.setup(datas,prior,tau,nwalkers)

	## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=100, progress=True)

	# Run 2000 fresh steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=200,progress=True)
	samples = b.mcmc.get_samples(sampler)
	b.mcmc.chain_statistics(sampler)
	b.mcmc.get_stats(sampler)

	lnp = sampler.get_log_prob()
	best = sampler.get_chain()[np.where(lnp==lnp.max())][0]

	xx = np.linspace(-.5,1.5,1000)
	fig,ax = plt.subplots(1)
	for i in range(len(datas)):
		ax.hist(datas[i],range=(-.5,1.5),bins=201,histtype='step',density=True)
		params = np.concatenate((best[:2],best[2+3*i:2+3*(i+1)]))
		yy = np.exp(b.likelihood.nosum_log_likelihood(params,xx,tau))
		ax.plot(xx,yy,color='k')
	plt.tight_layout()
	plt.savefig(os.path.join(fdir,'fig_constantepsilon.png'))
	plt.show()	

	## Check results
	truth = np.array((0.,1.,))
	target = np.array((.02,.02,))
	for i in range(len(datas)):
		truth = np.append(truth,np.array([sigmas[i],k1s[i],k2s[i]]))
		target = np.append(target,np.array([0.02,.1*k1s[i],.1*k1s[i]]))
	delta = np.abs(best-truth)/truth
	print(best)
	print(truth)
	print(delta)
	target = 0.1
	print(delta<target)
	assert np.all(delta < target)

if __name__ == '__main__':
	test_constantepsilon()