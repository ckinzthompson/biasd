# -*- coding: utf-8 -*-®
'''
PyQt trace plotter widget
'''
from PyQt5.QtWidgets import QWidget,QSizePolicy

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt

class trace_plotter(FigureCanvas):
	def __init__(self,parent=None):
		self.f, self.a = plt.subplots(1,figsize=(8,4))
		FigureCanvas.__init__(self,self.f)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		
		FigureCanvas.updateGeometry(self)
		self.prep_axis()
	
	def prep_axis(self):
		self.a.spines['right'].set_visible(False)
		self.a.spines['top'].set_visible(False)
		self.a.yaxis.set_ticks_position('left')
		self.a.xaxis.set_ticks_position('bottom')
		
		self.a.set_xlabel('Time (s)')
		self.a.set_ylabel('Signal')
		
		self.line = self.a.plot([],[])[0]
		self.plot_trace([0,1.],[0,1.])
		self.a.set_title('-/0')
		self.f.tight_layout()
		self.line.set_xdata([])
		self.line.set_ydata([])
		
		self.draw()
	
	def plot_trace(self,t,d,title=''):
		if np.ndim(d) == 1:
			if t is None:
				t = np.arange(d.size)
			self.line.set_xdata(t)
			self.line.set_ydata(d)
			self.a.set_xlim(np.min(t),np.max(t))
			deltad = 0.1*(np.max(d)-np.min(d))
			self.a.set_ylim(np.min(d) - deltad, np.max(d) + deltad)
			self.line.set_color('black')
			self.line.set_linewidth(1.)
			self.a.set_title(title)
			self.draw()
