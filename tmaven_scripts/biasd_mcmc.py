## tMAVEN gives `gui` and `maven`

import biasd as b
import numpy as np
import matplotlib.pyplot as plt
b.likelihood.use_python_numba_ll()

## get data
success,keep,y = maven.modeler.get_fret_traces()
tau = maven.prefs['plot.time_dt']

## clip data
data = np.concatenate([y[i] for i in range(len(y)) if keep[i]])
dclip = .5
keepclip = np.bitwise_and(data>-dclip,data<1.+dclip)
data = data[keepclip]

## initialize guess
kmeans = maven.modeler.cached_kmeans(data,2)
order = kmeans.mean.argsort()
mu = kmeans.mean[order]
std = np.sqrt(kmeans.var)[order]
guess = np.array((mu[0],mu[1],.5*(std[0]+std[1]),1./tau,1./tau))

## fit histogram
theta,covars = b.histogram.fit_histogram(data,tau,guess)
guess = theta.copy()


e1 = b.distributions.uniform(0.,.5)
e2 = b.distributions.uniform(.5,1)
sigma = b.distributions.loguniform(.01,.1)
k12 = b.distributions.loguniform(1e-2/tau,1e2/tau)
k21 = b.distributions.loguniform(1e-2/tau,1e2/tau)
prior = b.distributions.parameter_collection(e1,e2,sigma,k12,k21)



nwalkers = 20
sampler, initial_positions = b.mcmc.setup(data,prior,tau,nwalkers)

## Burn-in 500 steps and then remove them form the sampler, but keep the final positions
sampler, burned_positions = b.mcmc.burn_in(sampler, initial_positions, nsteps=20, progress=True)

# Run 2000 fresh steps starting at the burned-in positions.
sampler = b.mcmc.run(sampler,burned_positions,nsteps=100,progress=True)
samples = b.mcmc.get_samples(sampler)
b.mcmc.get_stats(sampler)


import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
oname = f"biasd_mcmc_samples_{current_time}.npy"
np.save(oname,samples)


## show plots
fig,ax = b.plot.mcmc_hist(data,tau,sampler)
plt.show()

# fig,ax=plt.subplots(5)
# for i in range(5):
# 	ax[i].hist(samples[:,i],bins=100)
# plt.show()