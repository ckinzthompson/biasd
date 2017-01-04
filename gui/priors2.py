# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup independent prior distributions
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import sys
#biasd_path = '/Users/colin/Desktop/20161220 biasd_release/biasd'
biasd_path = '../'
sys.path.append(biasd_path)
import biasd as b

from smd_loader import ui_loader

class distribution(QWidget):
	def __init__(self,dist_name):
		super(QWidget,self).__init__()
		self.dist_types = ['beta','empty','gamma','normal','uniform']
		self.dist_fxns = [b.distributions.beta,
			b.distributions.empty,
			b.distributions.gamma,
			b.distributions.normal,
			b.distributions.uniform
		]
		self.initialize(dist_name)

	def initialize(self,dist_name):
		self.hbox = QHBoxLayout()
		
		distlabel = QLabel(dist_name)
		
		self.prior_type = QComboBox()
		[self.prior_type.addItem(i) for i in self.dist_types]
		
		self.param1 = QLineEdit()
		self.param2 = QLineEdit()
		
		self.hbox.addStretch(1)
		self.hbox.addWidget(distlabel)
		self.hbox.addWidget(self.prior_type)
		self.hbox.addWidget(self.param1)
		self.hbox.addWidget(self.param2)
		
		self.setLayout(self.hbox)
		self.show()
	
	def get_distribution(self):
		dtype = str(self.prior_type.currentText())
		if dtype == 'empty':
			pp1 = 0.
			pp2 = 0.
		else:
			pp1 = float(self.param1.text())
			pp2 = float(self.param2.text())
		fxn = self.dist_fxns[self.dist_types.index(dtype)]
		return fxn(pp1,pp2)
	
class priors(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.priors_dists = [u'ϵ_1',u'ϵ_2',u'σ',u'k_1',u'k_2']
		self.initialize()

	def initialize(self):
		self.setWindowTitle('Set Priors')
		
		vbox = QVBoxLayout()
		hbox = QHBoxLayout()
		hbox.addStretch(1)
		
		bload = QPushButton("Load")
		bcheck = QPushButton("Check")
		bview = QPushButton("View")
		bset = QPushButton("Set")

		bload.clicked.connect(self.load_prior)
		bcheck.clicked.connect(lambda:self.check_dist(warn=True))
		bview.clicked.connect(self.view_dist)
		bset.clicked.connect(self.set_dist)
		
		[hbox.addWidget(i) for i in [bload,bcheck,bview,bset]]
		
		vbox.addStretch(1)
		vbox.addLayout(hbox)
		
		self.dists = [distribution(self.priors_dists[i]) for i in range(5)]
		[vbox.addWidget(self.dists[i]) for i in range(5)]
		
		self.update_dists()
		self.setLayout(vbox)
		self.setGeometry(200,200,500,300)
		self.show()

	def update_dists(self):
		try:
			p = self.parent().parent().priors
			for i,pp in zip(range(5),[p.e1,p.e2,p.sigma,p.k1,p.k2]):
				cb = self.dists[i].prior_type
				cb.setCurrentIndex(cb.findText(pp.name))
				self.dists[i].param1.setText(str(pp.parameters[0]))
				self.dists[i].param2.setText(str(pp.parameters[1]))
		except:
			QMessageBox.critical(self,"Priors are misformed","Priors are misformed. Cannot load them. Update them here.")

	def get_smd_filename(self):
		try:
			return self.parent().parent().filename
		except:
			return None
	
	def load_prior(self):
		try:
			if self.load.ui.filename == self.get_smd_filename():
				self.loader.setVisible(True)
				self.loader.raise_()
				return
		except:
			pass
		
		self.loader = ui_loader(self)
		self.loader.show()
	
	def select_callback(self,loc):
		self.loader.close()
		try:
			import h5py
			fn = self.get_smd_filename()
			f = h5py.File(fn,'r')
			g = f[loc]
			if g.attrs['description'] == 'BIASD parameter collection':
				p = b.smd.read.parameter_collection(g)
				self.parent().parent().priors = p
				self.update_dists()
				n,p = p.format()
				logstring = "Loaded Priors from "+fn+"[" + loc+ "]" + ''.join(["\n "+nn+" - "+ str(pp) for nn,pp in zip(n,p)])
				self.parent().parent().log.new(logstring)
				self.parent().parent().parent().statusBar().showMessage("Priors Loaded")
			else:
				QMessageBox.critical(self,"Not a parameter collection","This doesn't appear to be a parameter collection")
				f.close()

		except:
			try:
				f.close()
			except:
				pass
			QMessageBox.critical(self,"Couldn't load this collection","Couldn't load this from the HDF5 file as a parameter collection")
	
	def get_dists(self):
		try:
			bd = []
			dist_check = 0
			for i in range(5):
				bd.append(self.dists[i].get_distribution())
			dist_check = 1
			pc = b.distributions.parameter_collection(*bd)
			pc.check_dists()
			return pc
		except:
			if dist_check:
				QMessageBox.critical(self,
					"Out of Bounds",
					"Something is out of bounds in %s. Check support of those distribution parameters."%(''.join([self.priors_dists[i] +" " for i in range(5) if not bd[i].okay]))
				)
			else:
				QMessageBox.critical(self,
					"Something is Malformed ",
					"Something is Malformed in %s"%(self.priors_dists[i])
				)
	
	def check_dist(self,warn=True):
		params = self.get_dists()
		if not params is None:
			if params.okay:
				if warn:
					QMessageBox.information(self,
						"It's okay",
						"These Distributions are okay."
					)
				return params
		return None
		
	def view_dist(self):
		params = self.check_dist(warn=False)
		if not params is None:
			try:
				self.viewer.f.clf()
				self.viewer = None
			except:
				pass
			self.viewer = b.distributions.viewer(params)
				
	def set_dist(self):
		params = self.check_dist(warn=False)
		if not params is None:
			try:
				self.parent().parent().priors = params
				self.update_dists()
				self.parent().parent().parent().statusBar().showMessage("Priors Set")
				n,p = params.format()
				logstring = "Set Priors" + ''.join(["\n "+nn+" - "+ str(pp) for nn,pp in zip(n,p)])
				self.parent().parent().log.new(logstring)
			except:
				print 'set nothing'
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
			

class ui_priors(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = priors(self)
		self.setCentralWidget(self.ui)
		self.setGeometry(100,100,400,300)
		self.show()
	
	def closeEvent(self,event):
		try:
			self.ui.loader.close()
		except:
			pass

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_priors()
	sys.exit(app.exec_())
