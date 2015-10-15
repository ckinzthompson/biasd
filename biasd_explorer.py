from biasd import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import sys

class simtrace:
	@staticmethod
	def generate_states(rates,tlength):
		state = np.random.randint(rates.shape[0])
		trace = []
		states = []
		ttotal = 0.
		while 1:
			durations = np.random.exponential(scale=1./rates[:,:,None],size=(rates.shape[0],rates.shape[0],np.floor(1.1*tlength*rates.max())))
			path = np.empty(durations.shape[2])
			dwell = np.empty(durations.shape[2])
			for i in range(path.size - 1):
				path[i] = state
				p = durations[state,:,i].argmin()
				dwell[i] = durations[state,p,i]
				state = p
				if i == 0:
					#random starting observation time
					dwell[i] -= np.random.uniform(0,dwell[i])
				
				ttotal += dwell[i]
				if ttotal > tlength:
				#truncate to desired length
					dwell[i] -= (ttotal - tlength)
					path = path[:i+1]
					dwell = dwell[:i+1]
					states.append(path)
					trace.append(dwell)
					break
					
			if ttotal >= tlength:
				break
		return np.array((states,trace)).reshape((2,np.size(states)))

	@staticmethod
	def render_trace(trace,dt,emission):
		tc = trace[1].cumsum()
		tc = np.append(0,tc)
		tlength = tc[-1]
		ntypes = np.unique(trace[0])
		dd = np.zeros(np.floor(tlength/dt))
		dtime = np.linspace(dt,tlength,dd.size)
		for i in range(dtime.size):
			t1 = dtime[i]
			t0 = t1 - dt
			a,b = np.searchsorted(tc,[t0,t1])
			# print tc[a],tc[b]
			if tc[a-1] < t0 and tc[b] > t1:
				subtrace = trace[:,a-1:b].copy()
				subtrace[1,0] -= (t0 - tc[a-1])
				subtrace[1,-1] -= (tc[b]-t1)
			elif tc[a] == 0.:
				subtrace = trace[:,a:b].copy()
				subtrace[1,-1] -= (tc[b]-t1)
			elif tc[b] == 0.:
				subtrace = trace[:,a-1:b].copy()
				subtrace[1,0] -= (t0 - tc[a-1])

			jtimes = np.zeros(ntypes.size)
			for j in ntypes:
				jtimes[j] = subtrace[1][subtrace[0]==j].sum()
			dd[i] = (emission * (jtimes/jtimes.sum())).sum()
		dtime -= dt
		return dtime,dd
	
	def __init__(self,biasdtheta,frames,tau):
		e1,e2,sigma,k1,k2 = biasdtheta
		self.k = np.array(([0.,k1],[k2,0.]))
		self.emission = np.array((e1,e2))
		self.a = self.generate_states(self.k,frames*tau)
		self.dt = tau
		self.sigma = sigma
		self.x,self.y = self.render_trace(self.a,self.dt,self.emission)
		self.y += np.random.normal(scale=self.sigma,size=self.y.size)
		
def updateplot(traces,ax,index):
	i = traces[index]
	for a in ax:
		a.cla()
	sigs = i.posterior.mu[2]
	ax[0].axhspan(ymin=i.posterior.mu[0]-sigs,ymax=i.posterior.mu[0]+sigs,color='r',alpha=.2,zorder=-1)
	ax[0].axhspan(ymin=i.posterior.mu[1]-sigs,ymax=i.posterior.mu[1]+sigs,color='r',alpha=.2,zorder=-1)
	ax[0].plot(np.arange(i.data.size)*i.tau,i.data)
	ax[0].set_xlim(0,i.data.size*i.tau)
	ax[0].axhline(y=i.posterior.mu[0],color='r')
	ax[0].axhline(y=i.posterior.mu[1],color='r')
	hy,hx=ax[1].hist(i.data,bins=i.data.size**.5,normed=1,histtype='stepfilled',alpha=.6)[:2]
	x = np.linspace(-.5,1.5,1e2)
	y = np.exp(log_likelihood(i.posterior.mu,x,i.tau))
	ax[1].plot(x,y,color='k',lw=1.5)
	x = np.linspace(-.5,1.5,1e4)
	ax[2].plot(x,stats.p_gauss(x,i.posterior.mu[0],i.posterior.covar[0,0]**.5),color='b',lw=1.5)
	ax[2].fill_between(x,stats.p_gauss(x,i.posterior.mu[0],i.posterior.covar[0,0]**.5),color='b',alpha=.6)
	ax[2].plot(x,stats.p_gauss(x,i.posterior.mu[1],i.posterior.covar[1,1]**.5),color='g',lw=1.5)
	ax[2].fill_between(x,stats.p_gauss(x,i.posterior.mu[1],i.posterior.covar[1,1]**.5),color='g',alpha=0.6)
	x = np.linspace(1e-3,.15,1e4)
	ax[3].plot(x,stats.p_gauss(x,i.posterior.mu[2],i.posterior.covar[2,2]**.5),color='b',lw=1.5)
	ax[3].fill_between(x,stats.p_gauss(x,i.posterior.mu[2],i.posterior.covar[2,2]**.5),color='b',alpha=.6)
	x = np.linspace(1e-4,40,1e4)
	ax[4].plot(x,stats.p_gauss(x,i.posterior.mu[3],i.posterior.covar[3,3]**.5),color='b',lw=1.5)
	ax[4].fill_between(x,stats.p_gauss(x,i.posterior.mu[3],i.posterior.covar[3,3]**.5),color='b',alpha=0.6)
	ax[4].plot(x,stats.p_gauss(x,i.posterior.mu[4],i.posterior.covar[4,4]**.5),color='g',lw=1.5)
	ax[4].fill_between(x,stats.p_gauss(x,i.posterior.mu[4],i.posterior.covar[4,4]**.5),color='g',alpha=0.6)
	ax[4].axvline(x=i.posterior.mu[3],color='b',lw=1.5)
	ax[4].axvline(x=i.posterior.mu[4],color='g',lw=1.5)
	ax[4].axvline(x=1./i.tau,color='r',lw=1.5)
	ax[4].set_xscale('log')
	for a in ax[1:]:
		a.set_yscale('log')
		a.set_ylim(ymin=1e-2)
	ax[1].set_ylim(ymin=1./i.data.size)
	ax[0].set_title(str(index+1)+" / "+str(len(traces)))	
	plt.draw()

class Index:
	def __init__(self,traces,axes):
		self.traces = traces
		self.nmax = len(traces)
		self.ind = 0
		self.ax = ax
		self.update()
		self.simulations = 0
		self.simdata = self.traces[self.ind].data
		
	def last(self):
		self.jump(self.nmax)
	def first(self):
		self.jump(-self.nmax)
		
	def jump(self,inc):
		self.ind += inc
		if self.ind >= self.nmax:
			self.ind = self.nmax - 1
		if self.ind < 0:
			self.ind = 0
		self.update()
		
	def simulate(self):
		self.simulations += 1
		t = simtrace(self.traces[self.ind].posterior.mu,self.traces[self.ind].data.size,self.traces[self.ind].tau)
		self.simdata = np.append(self.simdata,t.y)
#		updateplot(self.traces,self.ax,self.ind)
		self.ax[0].plot(t.x+t.x.max()*self.simulations,t.y,color='purple')
		self.ax[0].axvline(x=t.x.max()*self.simulations,color='k',ls='--',lw=1.)
		self.ax[0].set_xlim(0.,(1.+self.simulations)*t.x.max())
#		self.ax[1].hist(self.simdata,bins=self.simdata.size**.5,normed=1,histtype='stepfilled',color='purple',alpha=0.3)
		plt.draw()
		
	def update(self):
		updateplot(self.traces,self.ax,self.ind)
		self.simulations = 0
		

argv = sys.argv
if len(argv) > 1:
	fname = sys.argv[1]
	d = dataset(analysis_fname=fname)
	d.load_analysis()

dtraces = []
mus = []
covars = []
for i in d.traces:
#	if i.data.size>100:
		if not i.posterior is None:
			dtraces.append(i)
			mus.append(i.posterior.mu)
			covars.append(i.posterior.covar)
mus = np.array(mus)
covars = np.array(covars)

##################
f,ax = plt.subplots(2,figsize=(8,8))
a0 = ax[0]
a1 = ax[1]

xe = np.linspace(0,1.,201)
xs = np.linspace(1e-3,.015,201)
xk = np.linspace(0,40,201)
# xorder = [xe,xe,xs,xk,xk]
# thetaname = [r'$\epsilon_1$',r'$\epsilon_2$',r'$\sigma$',r'$k_1$',r'$k_2$']

xe1,xe2 =np.meshgrid(xe,xe)
xk1,xk2 = np.meshgrid(xk,xk)
ze = np.log(stats.p_gauss(xe1[None,...],mus[:,0,None,None],covars[:,0,0,None,None]**.5)) + np.log(stats.p_gauss(xe2[None,...],mus[:,1,None,None],covars[:,1,1,None,None]**.5))
print 1
zk = np.log(stats.p_gauss(xk1[None,...],mus[:,3,None,None],covars[:,3,3,None,None]**.5)) + np.log(stats.p_gauss(xk2[None,...],mus[:,4,None,None],covars[:,4,4,None,None]**.5))
print 2
ze = np.sum(np.exp(ze),axis=0)
zk = np.sum(np.exp(zk),axis=0)

print 3
a0.pcolor(xe1,xe2,np.log(ze))
print 4
a1.pcolor(xk1,xk2,np.log(zk))
print 5
plt.show()


##################

f,ax=plt.subplots(5,figsize=(10,8),dpi=100,gridspec_kw=dict(height_ratios=[4,1,2,2,2]))
f.canvas.set_window_title('BIASD Explorer - '+fname)
callback = Index(dtraces,ax)

def key_selector(event):
	if event.key == 'right':
		callback.jump(1)
	elif event.key =='left':
		callback.jump(-1)
	elif event.key =='up':
		callback.last()
	elif event.key =='down':
		callback.first()
	elif event.key =='d':
		callback.jump(10)
	elif event.key == 'a':
		callback.jump(-10)
	elif event.key == 't':
		callback.simulate()

plt.connect('key_press_event', key_selector)
plt.tight_layout()
plt.show()
