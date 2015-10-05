from Tkinter import *
import tkFileDialog,tkMessageBox,tkSimpleDialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='ignore')
from biasd import *


class BIASDapp(Tk):
	#Frames Adapted from http://pythonprogramming.net/dashboard/#tab_guis
	def __init__(self,*args,**kwargs):
		Tk.__init__(self,*args,**kwargs)
		Tk.wm_title(self,"BIASD")
		#~ Tk.geometry(self,"800x300")
		
		self.bwindow = Frame(self)
		self.bwindow.pack(side="top", fill="both", expand = True)
		
		self.protocol(name="WM_DELETE_WINDOW",func=quit)
		
		self.frames = {}
		for F in [Trajectories,Priors,Analysis,MainPage]:
			frame = F(self.bwindow,self)
			self.frames[F] = frame
			frame.grid(row=0,column=0,sticky='nsew')
		self.show_frame(MainPage)
		self.update_prior()

	def show_frame(self, cont):
		frame = self.frames[cont]
		if frame.__class__.__name__ == "Trajectories":
			frame.updateplot()
		#print frame.__class__.__name__
		frame.tkraise()
	
	def reset_frame(self,cont):
		self.frames[cont] = cont(self.bwindow,self)
		self.frames[cont].grid(row=0,column=0,sticky='nsew')
	
	def update_prior(self):
		global data
		if not data.prior is None:
			frame = self.frames[Priors]
			dictpriortypeoptions = {'uniform':"Uniform (x,x)",'normal':"Normal (mu,sigma)",'gamma':"Gamma (alpha,beta)",'beta':"Beta (alpha,beta)"}

			for p,pd in zip([frame.pe1,frame.pe2,frame.psig,frame.pk1,frame.pk2],data.prior.list):
				p.set(dictpriortypeoptions[pd.type])
			for p1,p2,pd in zip([frame.pp1_e1,frame.pp1_e2,frame.pp1_sig,frame.pp1_k1,frame.pp1_k2],[frame.pp2_e1,frame.pp2_e2,frame.pp2_sig,frame.pp2_k1,frame.pp2_k2],data.prior.list):
				p1.set(pd.p1),p2.set(pd.p2)
	
	def update_analysis(self):
		global data
		try:
			if not data.ensemble_result is None:
				frame = self.frames[Analysis]
				frame.text_output.delete(1.0,END)
				frame.text_output.insert(1.0,data.ensemble_result.report())
				frame.updateplot()
		except:
			pass

class MainPage(Frame):
	def __init__(self,parent,controller):
		Frame.__init__(self,parent)
		global data
		global integralflag
		global integrandlibraryloc
		
		self.analysis_name = StringVar()
		self.data_name = StringVar()
		self.fmt = StringVar()
		self.tau = DoubleVar()
		self.integral_name = StringVar()
		
		def callback(*args):
			global data
			data.fmt = self.fmt.get()
			data.tau = self.tau.get()
			data.traces = []
			data.load_data()
		self.fmt.set('2D-NxT')
		self.fmt.trace("w", callback)
		self.tau.set(data.tau)
		self.tau.trace("w", callback)
		
		if integralflag:
			self.integral_name.set('Integrand: .C - '+integrandlibraryloc)
		else:
			self.integral_name.set('Integrand: Python')
		

		bframe = LabelFrame(self,borderwidth=0)
		bframe.grid(row=0,column=0,sticky='w')
		button1 = Button(self,text='Main Page',borderwidth=0,command=lambda:controller.show_frame(MainPage))
		button2 = Button(self,text='Trajectories',borderwidth=0,command=lambda:controller.show_frame(Trajectories))
		button3 = Button(self,text='Priors',borderwidth=0,command=lambda:controller.show_frame(Priors))
		button4 = Button(self,text='Analysis',borderwidth=0,command=lambda:controller.show_frame(Analysis))
		for i,b in zip(range(4),[button1,button2,button3,button4]):
			b.grid(in_=bframe,row=1,column=i,sticky=W)
		
		button1.config(background='#ADD8C7')
		
		def open_analysis(self,controller):
			global data
			proceed = data.analysis_fname is None
			if not proceed:
				proceed = tkMessageBox.askyesno(title="Discard Current Analysis?",message="Do you want to discard analysis entitled \n\"%s\"" %data.analysis_fname,default=tkMessageBox.NO)
			if proceed:
				fprompt = tkFileDialog.Open(defaultextension='biasd',filetypes=[('BIASD','.b'),('All','*')],initialdir='./',title='Load Analysis')
				fname = fprompt.show()
				if fname != "":
					try:
						data2 = dataset(analysis_fname=fname)
						data2.load_analysis()
						Tk.wm_title(controller,'BIASD - '+data2.analysis_fname.split('/')[-1])
						self.analysis_name.set('Analysis: '+fname)
						self.text_notes.delete(1.0,END)
						self.text_notes.insert(END,data2.title)
						self.tau.set(data2.tau)
						self.fmt.set(data2.fmt)
						data = data2
						controller.update_prior()
						controller.update_analysis()
						self.data_name.set("Data: "+data.data_fname)
					except:
						tkMessageBox.showwarning("Load Analysis","Cannot load \n%s" % fname)
		
		def new_analysis(self,controller):
			global data
			proceed = data.analysis_fname is None
			if not proceed:
				proceed = tkMessageBox.askyesno(title="Discard Current Analysis?",message="Do you want to discard analysis entitled \n\"%s\"" %data.analysis_fname,default=tkMessageBox.NO)
			if proceed:
				Tk.wm_title(controller,'BIASD - New Analysis')
				data = dataset(tau=1.)
				for F in [MainPage,Trajectories,Priors,Analysis]:
					controller.reset_frame(F)
				controller.show_frame(MainPage)
				self.analysis_name.set('No Saved Analysis')
				
		def save_analysis(self,controller):
			global data
			if self.analysis_name.get() == 'No Saved Analysis':
				if not data.data_fname is None:
					defaultname = data.data_fname.split('/')[-1][:-4]+".b"
				else:
					defaultname = ""
				fprompt = tkFileDialog.SaveAs(defaultextension='b',filetypes=[('BIASD','.b'),('All','*')],initialdir='./',title='Save Analysis',initialfile=defaultname)
				fname = fprompt.show()
				if fname != "":
					try:
						print fname
						data.analysis_fname = fname
						data.title = self.text_notes.get(1.0,END)
						data.save_analysis()
						self.analysis_name.set("Analysis: "+fname)
						Tk.wm_title(controller,'BIASD - '+data.analysis_fname.split('/')[-1])
					except:
						print 'Error Saving Analysis File'
						data.analysis_fname = None
			else:
				try:
					data.title = self.text_notes.get(1.0,END)
					data.save_analysis()
					print 'Saved to '+data.analysis_fname
				except:
					print "Error Saving Analysis File"
		
		def load_data(self,controller):
			global data
			fprompt = tkFileDialog.Open(defaultextension='dat',filetypes=[('Data File','.dat'),('All','*')],initialdir='./',title='Load Trajectories')
			fname = fprompt.show()
			if fname != "":
				try:
					data.data_fname = fname
					if data.tau is None:
						data.tau = 1.
					data.traces = []
					data.load_data()
					self.data_name.set("Data: "+fname)
				except:
					print 'Error Loading Data File - Check that it is tab-sepearated?'
					
		def get_tau(self,controller):
			global data
			ttau = tkSimpleDialog.askfloat('Tau','Enter length of each datapoint (tau) in seconds',minvalue=1e-300,initialvalue=data.tau)
			if not ttau is None:
				self.tau.set(ttau)
				data.tau = ttau
		
		def load_cintegrand(self):
			global integralflag
			global integrandlibraryloc
			fprompt = tkFileDialog.Open(defaultextension='so',filetypes=[('Library','.so'),('All','*')],initialdir='./',title='Location of C Integrand Library')
			fname = fprompt.show()
			if fname != "":
				integralflag = load_c_integral(fname)
				if integralflag == 1:
					integrandlibraryloc = fname
					self.integral_name.set('Integrand: .C - '+integrandlibraryloc)
				else:
					self.integral_name.set('Integrand: Python')
			
		def speed_wrapper():
			global integralflag
			if integralflag:
				test_speed(30)
			else:
				test_speed(3)

		analysisframe = Frame(self)
		integralframe = Frame(self)
		button_new = Button(self,text='New Analysis',command=lambda:new_analysis(self,controller))
		button_load = Button(self,text='Load Analysis',command=lambda:open_analysis(self,controller))
		button_loadintegrand = Button(self,text='Load Integrand',command=lambda:load_cintegrand(self))
		button_test_speed = Button(self,text='Test Speed',command=speed_wrapper)
		label_analysis_name = Label(self,textvariable=self.analysis_name)
		
		analysisframe.grid(row=1,column=0,sticky=EW,pady=10,padx=5)
		button_new.grid(in_=analysisframe,row=0,column=0,sticky=W)
		button_load.grid(in_=analysisframe,row=0,column=1,sticky=W,columnspan=1)
		button_save = Button(self,text='Save Analysis',command=lambda:save_analysis(self,controller))
		button_save.grid(in_=analysisframe,row=0,column=2,sticky=W)

		self.analysis_name.set('No Saved Analysis')
		self.data_name.set('Data: None')
		
		
		dataframe = Frame(self)
		noteframe = Frame(self)

		
		button_load_data = Button(self,text="Load Data",command=lambda:load_data(self,controller))
		label_data_fname = Label(self,text='No Data Loaded',textvariable=self.data_name)
		label_integral = Label(self,textvariable=self.integral_name)
		label_format = Label(self,text='Data Format',pady=15)
		option_format = OptionMenu(self,self.fmt,*['2D-NxT','2D-TxN','1D'])
		button_tau = Button(self,text='Set Tau',command=lambda:get_tau(self,controller))
		label_tau = Label(self,textvariable=self.tau,pady=15)
		label_tau2 = Label(self,text="Tau (sec) :",pady=15)
		label_notes = Label(self,text='Notes')
		scroll_notes = Scrollbar(self)
		self.text_notes = Text(self,height=25,width=35)
		
		dataframe.grid(row=2,sticky=EW,pady=10,padx=5)
		button_load_data.grid(in_=dataframe,row=2,column = 0,sticky=EW,columnspan=1)
		

		label_format.grid(in_=dataframe,row=0,column = 0,columnspan=2,sticky=W)
		option_format.grid(in_=dataframe,row=0,column = 2)
		button_tau.grid(in_=dataframe,row=1,column = 2,columnspan=1,sticky=EW)
		label_tau.grid(in_=dataframe,row=1,column = 1,sticky=E)
		label_tau2.grid(in_=dataframe,row=1,column = 0,sticky=W)
		
		noteframe.grid(row=1,column=1,rowspan=4,sticky=EW,pady=10,padx=5)
		label_notes.grid(in_=noteframe,row=0,column =0,columnspan=1)
		self.text_notes.grid(in_=noteframe,row=1,column=0,sticky='NESW')
		scroll_notes.grid(in_=noteframe,row=1,column=1,sticky='NESW')
		scroll_notes.config(command=self.text_notes.yview)
		self.text_notes.config(yscrollcommand=scroll_notes.set)
			
		label_data_fname.grid(row=6,column = 0,columnspan=2,sticky=W)
		label_analysis_name.grid(row=5,column=0,sticky=W,columnspan=2)
		label_integral.grid(row=7,column=0,columnspan=2,sticky=W)

		integralframe.grid(row=8,column=0,sticky=EW)
		button_loadintegrand.grid(in_=integralframe,row=0,column=0,sticky=W)
		button_test_speed.grid(in_=integralframe,row=0,column=1,sticky=W)


class Trajectories(Frame):
	def __init__(self,parent,controller):
		Frame.__init__(self,parent)
		
		global data
		
		bframe = Frame(self)
		bframe.grid(row=0,column=0,sticky='w')
		button1 = Button(self,text='Main Page',borderwidth=0,command=lambda:controller.show_frame(MainPage))
		button2 = Button(self,text='Trajectories',borderwidth=0,command=lambda:controller.show_frame(Trajectories))

		button3 = Button(self,text='Priors',borderwidth=0,command=lambda:controller.show_frame(Priors))
		button4 = Button(self,text='Analysis',borderwidth=0,command=lambda:controller.show_frame(Analysis))
		for i,b in zip(range(4),[button1,button2,button3,button4]):
			b.grid(in_=bframe,row=1,column=i,sticky=W)
			
		button2.config(background='#ADD8C7')

		self.fig,self.ax = plt.subplots(2,figsize=(8,6))
		self.fig.subplots_adjust(bottom=0.225)
		self.trace_index = 0
		
		self.ax[1].set_title('Histogram of All Traces',fontsize=14)
		self.ax[1].set_ylabel('Probability',fontsize=14)
		self.ax[1].set_xlabel(r'$Intensity$',fontsize=14)
		
		self.canvas = FigureCanvasTkAgg(self.fig,self)
		self.canvas.show()
		self.canvas.get_tk_widget().grid(row=1,column=0,sticky='ew')
		self.fig.set_visible(False)
		#~ toolbar = NavigationToolbar2TkAgg(self.canvas,self)
		#~ toolbar.update()
		#~ toolbar.grid(row=3,column=0,sticky='ew')
		
		def jump_change(self,dx):
			self.trace_index += dx
			self.updateplot()
		
		navframe = Frame(self)
		button_start = Button(self,text=' | ',command=lambda: jump_change(self,-np.size(data.traces)))
		button_end = Button(self,text=' | ',command=lambda: jump_change(self,+np.size(data.traces)))
		button_fwd1 = Button(self,text=' > ',command=lambda: jump_change(self,1))
		button_rev1 = Button(self,text=' < ',command=lambda: jump_change(self,-1))
		button_fwd10 = Button(self,text=' >>',command=lambda: jump_change(self,10))
		button_rev10 = Button(self,text='<< ',command=lambda: jump_change(self,-10))
		navframe.grid(row=2,column=0)
		button_start.grid(in_=navframe,row=0,column=0)
		button_end.grid(in_=navframe,row=0,column=5)
		button_fwd1.grid(in_=navframe,row=0,column=3)
		button_rev1.grid(in_=navframe,row=0,column=2)
		button_fwd10.grid(in_=navframe,row=0,column=4)
		button_rev10.grid(in_=navframe,row=0,column=1)
		
		controller.bind('<Right>',lambda x:jump_change(self,1))
		controller.bind('<Left>',lambda x:jump_change(self,-1))		
		controller.bind('<Shift-Right>',lambda x:jump_change(self,10))		
		controller.bind('<Shift-Left>',lambda x:jump_change(self,-10))		

	def updateplot(self):
		global data
		self.fig.set_visible(True)
		try:
			if self.trace_index >= np.size(data.traces):
				self.trace_index  = np.size(data.traces) - 1
			elif self.trace_index < 0:
				self.trace_index = 0
			
			d = data.traces[self.trace_index]
			xmax = d.data.size
		
			a0 = self.ax[0]
			a0.cla()
			self.ax[1].cla()
			self.ax[1].hist(data.data[1],bins=data.data[1].size**.5,normed=1,alpha=.3,histtype='stepfilled',color='k')
			self.ax[1].set_xlim(data.data[1].min(),data.data[1].max())
			self.ax[1].set_title('Histogram of All Traces',fontsize=14)
			self.ax[1].set_ylabel('Probability',fontsize=14)
			self.ax[1].set_xlabel('Intensity',fontsize=14)
			
			a0.set_title('Trace '+ str(self.trace_index+1) + '/' + str(np.size(data.traces)),fontsize=14)
			tt = np.linspace(0,xmax*data.tau,xmax)
			a0.plot(tt,d.data,'k',lw=1)
			a0.set_xlabel('Time (s)',fontsize=14)
			a0.set_ylabel('Intensity',fontsize=14)
			a0.set_xlim(0,tt.max())
			a0.locator_params(nbins=8,axis='x')
			a0.locator_params(nbins=5,axis='y')
			
			self.fig.tight_layout()
			self.canvas.draw()
		except:
			pass


class Priors(Frame):
	def __init__(self,parent,controller):
		Frame.__init__(self,parent)

		bframe = Frame(self)
		bframe.grid(row=0,column=0,sticky='w')
		button1 = Button(self,text='Main Page',borderwidth=0,command=lambda:controller.show_frame(MainPage))
		button2 = Button(self,text='Trajectories',borderwidth=0,command=lambda:controller.show_frame(Trajectories))
		button3 = Button(self,text='Priors',borderwidth=0,command=lambda:controller.show_frame(Priors))
		button4 = Button(self,text='Analysis',borderwidth=0,command=lambda:controller.show_frame(Analysis))
		for i,b in zip(range(4),[button1,button2,button3,button4]):
			b.grid(in_=bframe,row=1,column=i,sticky=W)
		
		button3.config(background='#ADD8C7')
		
		priorsframe = LabelFrame(self,pady=5)
		priorsframe.grid(row = 1,sticky=E+W+N+S, column =0)
		
		ptopl_p1 = Label(self,text='Parameter 1')
		ptopl_p2 = Label(self,text='Parameter 2')
		ptopl_dist = Label(self,text='Distribution')
		
		ple1 = Label(self,text=u'\u03b51',justify=RIGHT)
		ple2 = Label(self,text=u'\u03b52',justify=RIGHT)
		plsig  = Label(self,text=u'\u03c3',justify=RIGHT)
		plk1 = Label(self,text='k1',justify=RIGHT)
		plk2 = Label(self,text=u'k2',justify=RIGHT)
		
		self.pe1 = StringVar()
		self.pe2 = StringVar()
		self.psig = StringVar()
		self.pk1 = StringVar()
		self.pk2 = StringVar()

		self.pp1_e1 = DoubleVar()
		self.pp2_e1 = DoubleVar()
		self.pp1_e2 = DoubleVar()
		self.pp2_e2 = DoubleVar()
		self.pp1_sig = DoubleVar()
		self.pp2_sig = DoubleVar()
		self.pp1_k1 = DoubleVar()
		self.pp2_k1 = DoubleVar()
		self.pp1_k2 = DoubleVar()
		self.pp2_k2 = DoubleVar()
		
		epp1_e1 = Entry(self,textvariable=self.pp1_e1)
		epp2_e1 = Entry(self,textvariable=self.pp2_e1)
		epp1_e2 = Entry(self,textvariable=self.pp1_e2)
		epp2_e2 = Entry(self,textvariable=self.pp2_e2)
		epp1_sig = Entry(self,textvariable=self.pp1_sig)
		epp2_sig = Entry(self,textvariable=self.pp2_sig)
		epp1_k1 = Entry(self,textvariable=self.pp1_k1)
		epp2_k1 = Entry(self,textvariable=self.pp2_k1)
		epp1_k2 = Entry(self,textvariable=self.pp1_k2)
		epp2_k2 = Entry(self,textvariable=self.pp2_k2)
		
		priortypeoptions = ["Uniform (x,x)","Normal (mu,sigma)","Gamma (alpha,beta)","Beta (alpha,beta)"]
		pme1 = OptionMenu(self,self.pe1,*priortypeoptions)
		pme2 = OptionMenu(self,self.pe2,*priortypeoptions)
		pmsig = OptionMenu(self,self.psig,*priortypeoptions)
		pmk1 = OptionMenu(self,self.pk1,*priortypeoptions)
		pmk2 = OptionMenu(self,self.pk2,*priortypeoptions)
		
		def checkprior(self,controller):
			global data
			cdist = [self.pe1.get(),self.pe2.get(),self.psig.get(),self.pk1.get(),self.pk2.get()]
			cparams = [[self.pp1_e1.get(),self.pp2_e1.get()],[self.pp1_e2.get(),self.pp2_e2.get()],[self.pp1_sig.get(),self.pp2_sig.get()],[self.pp1_k1.get(),self.pp2_k1.get()],[self.pp1_k2.get(),self.pp2_k2.get()]]
			cnames = [u'\u03b51',u'\u03b52',u'\u03c3','n',u'\u0394']
			cpos = range(5)
			for i in cpos:
				cdist[i] = cdist[i].split(' ')[0]
			temp_prior = biasddistribution(*[dist(cdist[i],*cparams[i]) for i in range(5)])
			if temp_prior.test_distributions():
				data.prior = temp_prior
				data.update()
			else:
				badlist = temp_prior.which_bad()
				for i in badlist:
					tkMessageBox.showerror("Prior Parameters Error","The parameters for %s are not valid parameters. Prior was not set."%i)
			
			
		checkpriorsbutton = Button(self,text='Check & Set Priors',command=lambda:checkprior(self,controller))
		checkpriorsbutton.grid(in_=priorsframe,row=5,column=4,sticky='e')
		button_plotpriors = Button(self,text='Plot Priors',command=self.updateplot)
		button_plotpriors.grid(in_=priorsframe,row=4,column=4,sticky='e')
		
		ptopl_dist.grid(in_=priorsframe,row=0,column=1,sticky='ew')
		ptopl_p1.grid(in_=priorsframe,row=0,column=2,sticky='ew')
		ptopl_p2.grid(in_=priorsframe,row=0,column=3,sticky='ew')
		ple1.grid(in_=priorsframe,row=1,column=0)
		ple2.grid(in_=priorsframe,row=2,column=0)
		plsig.grid(in_=priorsframe,row=3,column=0)
		plk1.grid(in_=priorsframe,row=4,column=0)
		plk2.grid(in_=priorsframe,row=5,column=0)
		pme1.grid(in_=priorsframe,row=1,column=1,sticky="ew")
		pme2.grid(in_=priorsframe,row=2,column=1,sticky="ew")
		pmsig.grid(in_=priorsframe,row=3,column=1,sticky="ew")
		pmk1.grid(in_=priorsframe,row=4,column=1,sticky="ew")
		pmk2.grid(in_=priorsframe,row=5,column=1,sticky="ew")
		epp1_e1.grid(in_=priorsframe,row=1,column=2,sticky="ew")
		epp2_e1.grid(in_=priorsframe,row=1,column=3,sticky="ew")
		epp1_e2.grid(in_=priorsframe,row=2,column=2,sticky="ew")
		epp2_e2.grid(in_=priorsframe,row=2,column=3,sticky="ew")
		epp1_sig.grid(in_=priorsframe,row=3,column=2,sticky="ew")
		epp2_sig.grid(in_=priorsframe,row=3,column=3,sticky="ew")
		epp1_k1.grid(in_=priorsframe,row=4,column=2,sticky="ew")
		epp2_k1.grid(in_=priorsframe,row=4,column=3,sticky="ew")
		epp1_k2.grid(in_=priorsframe,row=5,column=2,sticky="ew")
		epp2_k2.grid(in_=priorsframe,row=5,column=3,sticky="ew")
	
		self.fig,self.ax = plt.subplots(2,2,figsize=(8,4))
		
		self.canvas = FigureCanvasTkAgg(self.fig,self)
		self.canvas.show()
		self.canvas.get_tk_widget().grid(row=2,column=0,sticky='nsew')
		self.fig.set_visible(False)
		#~ toolbar = NavigationToolbar2TkAgg(self.canvas,self)
		#~ toolbar.update()
		#~ toolbar.grid(row=3,column=0,sticky='ew')
	
	def updateplot(self):
		global data
		self.fig.set_visible(True)
		nnlabel = [r'$\epsilon_1$',r'$\epsilon_2$',r'$\sigma$',r'k$_1$',r'k$_2$']
			
		((ax1,ax2),(ax3,ax4)) = self.ax
		for a in self.ax:
			a[0].cla()
			a[1].cla()
		ax = [ax1,ax1,ax2,ax3,ax3]
		color=['b','r','g','k','orange']
		
		xlim1 = [[],[]]
		xlim3 = [[],[]]
		for pd,a,c,lab in zip(data.prior.list,ax,color,nnlabel):
			if pd.type == 'uniform':
				x = np.linspace(pd.p1,pd.p2,1001)
			elif pd.type == 'normal':
				x = np.linspace(pd.p1-5.*pd.p2,pd.p1+5.*pd.p2,1001)
			elif pd.type == 'gamma':
				x = np.linspace(1e-6,special.gammaincinv(pd.p1,.99)/pd.p2,1001)
			elif pd.type == 'beta':
				x = np.linspace(.001,.999,1001)
			a.plot(x,pd.pdf(x),color=c,lw=1.5,label=lab)
			a.fill_between(x,pd.pdf(x),color=c,alpha=.6)
			a.legend(fontsize=8)
			a.set_yticks((),())
			a.locator_params(nbins=4,axis='x')
			if a == ax1:
				xlim1[0].append(x.min())
				xlim1[1].append(x.max())
			elif a == ax3:
				xlim3[0].append(x.min())
				xlim3[1].append(x.max())
			else:
				xlim2 = [x.min(),x.max()]
			
		xlim1[0] = np.min(xlim1[0])
		xlim1[1] = np.max(xlim1[1])
		xlim3[0] = np.min(xlim3[0])
		xlim3[1] = np.max(xlim3[1])
		y1max = np.max((data.prior.e1.pdf(data.prior.e1.mean()),data.prior.e2.pdf(data.prior.e2.mean())))*1.3
		y2max = data.prior.sigma.pdf(data.prior.sigma.mean())*1.3
		y3max = np.max((data.prior.k1.pdf(data.prior.k1.mean()),data.prior.k2.pdf(data.prior.k2.mean())))*1.3
		
		ax1.set_xlim(*xlim1)
		ax2.set_xlim(*xlim2)
		ax3.set_xlim(*xlim3)
		ax1.set_ylim(0.,y1max)
		ax2.set_ylim(0.,y2max)
		ax3.set_ylim(0.,y3max)
		
		global integralflag
		if integralflag:
			npoints = 1000
		else:
			npoints = 10
			
		hy,hx = np.histogram(data.data[1],bins=data.data[1].size**.5,normed=1)
		hxx = np.linspace(hx.min(),hx.max(),101)
		ax4.hist(data.data[1],bins=data.data[1].size**.5,normed=1,alpha=.6,histtype='stepfilled',color='b',label='data')
		ll = np.exp(log_likelihood(data.prior.random_theta(),hxx,data.tau))
		for i in range(npoints-1):
			ll += np.exp(log_likelihood(data.prior.random_theta(),hxx,data.tau))
		ll /= ll.sum() * (hxx[1]-hxx[0])
		ax4.plot(hxx,ll,color='k',lw=1.5,label='Marginal')
		ax4.fill_between(hxx,ll,color='k',alpha=.4)
		ax4.legend(fontsize=8)
		
		ax4.set_yticks((),())
		ax4.set_xlim(hx.min(),hx.max())
		ax4.set_ylim(0.,np.max((hy.max(),ll.max()))*1.3)
		
		ax4.locator_params(nbins=8,axis='x')
		#ax4.locator_params(nbins=5,axis='y')
		
		self.fig.tight_layout()
		self.canvas.draw()
		

class Analysis(Frame):
	def __init__(self,parent,controller):
		Frame.__init__(self,parent)
		
		
		self.keyplotstate = 0
		
		bframe = Frame(self)
		bframe.grid(row=0,column=0,sticky='w')
		button1 = Button(self,text='Main Page',borderwidth=0,command=lambda:controller.show_frame(MainPage))
		button2 = Button(self,text='Trajectories',borderwidth=0,command=lambda:controller.show_frame(Trajectories))
		button3 = Button(self,text='Priors',borderwidth=0,command=lambda:controller.show_frame(Priors))
		button4 = Button(self,text='Analysis',borderwidth=0,command=lambda:controller.show_frame(Analysis))
		for i,b in zip(range(4),[button1,button2,button3,button4]):
			b.grid(in_=bframe,row=1,column=i,sticky=W)
		
		button4.config(background='#ADD8C7')
		
		self.nproc = IntVar()
		self.nstates = IntVar()
		self.nproc.set(mp.cpu_count())
		self.nstates.set(5)
		
		analysisframe = Frame(self,pady=10)
		runframe = Frame(self)
		plotframe = Frame(self)
		outputframe = Frame(self)
		
		analysisframe.grid(row=1,sticky='ew')
		
		plotframe.grid(in_=analysisframe,row=0,column=1,sticky='ew')
		runframe.grid(in_=analysisframe,row=0,column=0,sticky='new')
		outputframe.grid(in_=runframe,row=2,column=0,columnspan=4,sticky='new')
		
		label_nproc = Label(self,text='# CPU')
		label_nstates = Label(self,text='Max States')
		option_nproc = OptionMenu(self,self.nproc,*np.linspace(1,mp.cpu_count(),mp.cpu_count(),dtype='i'))
		option_nstates = OptionMenu(self,self.nstates,*np.linspace(1,20,20,dtype='i'))
		self.button_run = Button(self,text='Run',command=lambda:run_analysis(self,controller))
		button_hierarchical = Button(self,text='Ensemble',command=lambda:hierarchical_analysis(self,controller))
		label_nproc.grid(in_=runframe,row=0,column=0,sticky='wn')
		label_nstates.grid(in_=runframe,row=0,column=1,columnspan=2,sticky='wn')
		option_nproc.grid(in_=runframe,row=1,column=0,sticky='wn')
		option_nstates.grid(in_=runframe,row=1,column=1,sticky='nw')
		
		def run_analysis(self,controller):
			global data
			postcount = 0
			for tracei in data.traces:
				if not tracei.posterior is None:
					postcount +=1
			
			if not data.ensemble_result is None:
				proceed = tkMessageBox.askyesno(title="Discard Current Posteriors?",message="Do you want to discard the current "+str(postcount)+" posteriors?",default=tkMessageBox.NO)
			else:
				proceed = 1
			
			if proceed and np.size(data.traces) > 0 and data.tau > 0 and data.prior.complete:
				data.run_laplace(nproc=self.nproc.get())
				postcount = 0
				for tracei in data.traces:
					if not tracei.posterior is None:
						postcount +=1
				self.text_output.delete(1.0,END)
				self.text_output.insert(1.0,"Approx. Posteriors for:\n"+str(postcount) + " / " + str(data.n_traces))
				
		def hierarchical_analysis(self,controller):
			global data
			if not data.ensemble_result is None:
				proceed = tkMessageBox.askyesno(title="Discard Current Posteriors?",message="Do you want to discard the current ensemble result?",default=tkMessageBox.NO)
			else:
				proceed = 1
			
			if proceed and np.size(data.traces) > 0 and data.tau > 0 and data.prior.complete:
				data.variational_ensemble(nstates=self.nstates.get(),nproc=self.nproc.get())
				self.text_output.delete(1.0,END)
				self.text_output.insert(1.0,data.ensemble_result.report())
				self.updateplot()
		
		
		self.button_run.grid(in_=runframe,row=1,column=2,sticky='nw')
		button_hierarchical.grid(in_=runframe,row=1,column=3,sticky='nw')
		
		label_output = Label(self,text='Output')
		self.text_output = Text(self,height=35,width=30)
		scroll_output = Scrollbar(self)
		
		label_output.grid(in_=outputframe,row=0,column =0,columnspan=1,sticky='w')
		self.text_output.grid(in_=outputframe,row=1,column=0,sticky='NESW')
		scroll_output.grid(in_=outputframe,row=1,column=1,sticky='NESW')
		scroll_output.config(command=self.text_output.yview)
		self.text_output.config(yscrollcommand=scroll_output.set)
		
		#self.fig,self.ax = plt.subplots(2,2,figsize=(5,6))
		self.fig,self.ax = plt.subplots(2,figsize=(5,6),gridspec_kw={'height_ratios':[1,4]})
		self.canvas = FigureCanvasTkAgg(self.fig,self)
		self.canvas.show()
		self.canvas.get_tk_widget().grid(in_=plotframe,row=0,column=0,sticky='nsew')
		self.fig.set_visible(False)
		
		controller.bind('<Key>',self.keysave)
		#~ toolbar = NavigationToolbar2TkAgg(self.canvas,self)
		#~ toolbar.update()
		#~ toolbar.grid(in_=plotframe,row=1,column=0,sticky='ew')
		
	def keysave(self,event):
		#this will work for now...
		if np.any(event.keysym == np.arange(1,10).astype('S2')) and self.keyplotstate:
			self.updateplot(n=int(event.keysym))
			self.keyplotstate = 0
		elif event.keysym =='s' or event.keysym == 'S':
			self.saveplots()
			self.keyplotstate = 0
		elif event.keysym == 'm' or event.keysym == 'M':
			self.updateplot()
			self.keyplotstate = 0
		#Control
		elif event.keycode == 37 or event.keycode == 109 :
			self.keyplotstate = 1
		else:
			self.keyplotstate = 0
		
		
	def saveplots(self):
		global data
		try:
			self.fig.savefig(data.analysis_fname[:-2]+"_ensemble.pdf")
			# self.fig.savefig(data.analysis_fname[:-4]+"_ensemble.png",dpi=300)
			f = open(data.analysis_fname[:-2]+"_ensemble.txt",'w')
			f.write(self.text_output.get(1.0,END))
			f.close()
		except:
			print "Ensemble Result Error"
	
	def updateplot(self,n=None):
		global data
		
		self.fig.set_visible(True)
		nnlabel = [r'$\epsilon_1$',r'$\epsilon_2$',r'$\sigma$',r'k$_1$',r'k$_2$']

		#Tableau 10 Medium
		colors = [	[0.4470588235294118, 0.6196078431372549, 0.807843137254902],
					[0.403921568627451, 0.7490196078431373, 0.3607843137254902],
					[0.9294117647058824, 0.4, 0.36470588235294116],
					[1.0, 0.6196078431372549, 0.2901960784313726],
					[0.6588235294117647, 0.47058823529411764, 0.43137254901960786],
					[0.9294117647058824, 0.592156862745098, 0.792156862745098],
					[0.6784313725490196, 0.5450980392156862, 0.788235294117647],
					[0.803921568627451, 0.8, 0.36470588235294116],
					[0.6352941176470588, 0.6352941176470588, 0.6352941176470588],
					[0.42745098039215684, 0.8, 0.8549019607843137]]

		ax1 = self.ax[0]
		ax4 = self.ax[1]
		
		klist = np.array((),dtype='i')
		evidences = np.array((),dtype='f')
		for er in data._ensemble_results:
			klist = np.append(klist,er[0].size)
			evidences = np.append(evidences,er[-2][-1][1])
		
		er = data.ensemble_result
		if not n is None and np.any(data._lb_states[0] == n):
			er = ensemble(data.ensemble_result.x,data._ensemble_results[int(data._lb_states[2,n-1])])
			plot_marginals(data._ensemble_results[int(data._lb_states[2,n-1])])
		
		ax1.cla()
		ax1.scatter(klist,evidences,s=5,color='k')
		ax1.plot(data._lb_states[0],data._lb_states[1],color='k',lw=1.5)
		ax1.axvline(x=er.alpha.size,color='r')
		ax1.set_xticks(data._lb_states[0])
		ax1.set_xlim(xmin=0)
		if data._lb_states.shape[0] > 1:
			ydelta = data._lb_states[1].max() - data._lb_states[1].min()
			ax1.set_ylim(data._lb_states[1].min()-np.abs(ydelta*.1),data._lb_states[1].max()+np.abs(ydelta*.1))
		ax1.set_xlabel('States')
		ax1.set_ylabel('Lower Bound',labelpad=15)
		ax1.set_yticks((),())

		
		global integralflag
		if integralflag:
			npoints = 1000
		else:
			npoints = 10
		
		
		ax4.cla()
		
		hxx = data._histograms[er.alpha.size-1][0][2]
		for k in range(er.alpha.size):
			ax4.plot(hxx,data._histograms[er.alpha.size-1][k][1],color=colors[k])
			ax4.fill_between(hxx,data._histograms[er.alpha.size-1][k][1],label=str(k+1),color=colors[k],alpha=.3)
			ax4.plot(hxx,data._histograms[er.alpha.size-1][k][3],color=colors[k],lw=1.5,label=str(k+1))
			
		ax4.legend(fontsize=8)
		
		ax4.set_yticks((),())
		#ax4.set_xlim(hx.min(),hx.max())
		ax4.set_xlim(-.4,1.4)
		ax4.set_ylim(0.,data._histograms[0][0][1].max()*1.3)
		ax4.set_ylabel('Data Frequency',labelpad=15)
		ax4.set_xlabel('Signal Units')
		ax4.locator_params(nbins=12,axis='x')
		#ax4.locator_params(nbins=5,axis='y')
		
		ax4.plot(data._histograms[0][0][2],data._histograms[0][0][1],color='k',lw=1.5,alpha=.8)
		
		self.fig.tight_layout()
		self.canvas.draw()

		self.text_output.delete(1.0,END)
		self.text_output.insert(1.0,er.report())

		recordout = "\nLower Bounds\nStates Best"
		for i in range(data._lb_states.shape[1]):
			recordout += "\n"+str(i+1)+" "+str(data._lb_states[1,i])
		self.text_output.insert(END,recordout)

###Initialize with some personal settings
#data = dataset(tau=1.,prior=prior_generic_distribution)
data = dataset(tau=.05,prior=prior_personal_distribution)

###Try here first
integrandlibraryloc = './integrand_gsl.so'
integralflag = load_c_integral(integrandlibraryloc)

#Start up the GUI
app = BIASDapp()
app.mainloop()

