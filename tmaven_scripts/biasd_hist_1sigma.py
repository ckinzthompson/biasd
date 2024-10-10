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

## print results
labels = ['e1','e2','s ','k1','k2']
for i in range(len(labels)):
	print(f'{labels[i]}: {theta[i]:.6f} +/- {np.sqrt(covars[i,i]):.6f}')

## show plots
fig,ax = b.plot.likelihood_hist(data,tau,theta)
plt.show()
