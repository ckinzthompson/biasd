import simulate_singlemolecules as ssm
import matplotlib.pyplot as plt
import numpy as np
import biasd as b
import corner
import pytest
import os

fdir = os.path.dirname(os.path.abspath(__file__))

def test_titration():
	b.likelihood.use_python_numba_ll_2sigma()

	#### Simulate data
	ka = 1.0 # uM^-1 s^-1
	kd = 1 # s^-1
	concs = np.array([0.01,.03,0.1,.3,1.,3.,10.,30.,100.]) ## uM
	k1s = ka*concs
	k2s = kd + 0*concs
	nconcs = concs.size

	datas = []
	for i in range(nconcs):
		rates = np.zeros((2,2))
		rates[0,1] = k1s[i]
		rates[1,0] = k2s[i]
		emissions = np.array((0.,1.))
		noise = 0.05
		nframes = 100
		dt = 1.
		nmol = 10

		data = ssm.simulate_ensemble(rates,emissions,noise,nframes,dt,nmol)
		datas.append(data.flatten())

		# plt.hist(data.flatten(),range=(-.5,1.5),bins=201,histtype='step')
	# plt.show()


	#### Analyze

	## SETUP
	tau = 1.0

	pe1 = b.distributions.normal(0.,.01)
	pe2 = b.distributions.normal(1.,.01)
	psigma1 = b.distributions.loguniform(.01,.1)
	psigma2 = b.distributions.loguniform(.01,.1)
	pka = b.distributions.loguniform(1.,30.)
	pkd = b.distributions.loguniform(1.,30.)
	prior = b.titration.collection_titration(pe1,pe2,psigma1,psigma2,pka,pkd)

	nwalkers = 20
	sampler, initial_positions = b.titration.setup(datas,concs,prior,tau,nwalkers)

	## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
	sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=100, progress=True)

	# Run 2000 fresh steps starting at the burned-in positions.
	sampler = b.mcmc.run(sampler,burned_positions,nsteps=100,progress=True)
	samples = b.mcmc.get_samples(sampler)
	b.mcmc.chain_statistics(sampler)
	b.mcmc.get_stats(sampler)

	## make plot
	fig,ax = plt.subplots(1,)
	cc = 10.**np.linspace(np.log10(concs.min()/5.),np.log10(concs.max()*5.),1000)
	# f2 = ka*concs/(ka*cc+kd)
	f2 = samples[:,4,None]*cc[None,:]/(samples[:,4,None]*cc[None,:]+samples[:,5,None])
	avg = samples[:,0,None]*(1.-f2) + samples[:,1,None]*f2
	for i in range(avg.shape[0]):
		ax.semilogx(cc,avg[i],color='tab:blue',alpha=.05)
	ax.semilogx(concs,np.array([datas[i].mean() for i in range(nconcs)]),'o',color='k')
	
	lnp = sampler.get_log_prob()
	best = sampler.get_chain()[np.where(lnp==lnp.max())][0]
	print(best.shape)
	
	f2 = best[4]*cc/(best[4]*cc+best[5])
	avg = best[0]*(1.-f2) + best[1]*f2
	ax.semilogx(cc,avg,color='tab:red')
	
	plt.tight_layout()
	plt.savefig(os.path.join(fdir,'fig_titration.png'))
	plt.close()

	## Check results
	truth = np.array((0.,1.,.05,.05,1.,1.))
	delta = np.abs(best-truth)
	target = np.array((.01,.01,.01,.01,.1,.1))
	print(best)
	print(truth)
	print(delta)
	print(delta<target)
	assert np.all(delta < target)

if __name__ == '__main__':
	test_titration()