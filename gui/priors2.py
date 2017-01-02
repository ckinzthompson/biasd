# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup independent prior distributions
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QGridLayout, QLabel, QLineEdit
from PyQt5.QtGui import QStandardItemModel, QStandardItem


class distribution(QWidget):
	def __init__(self,dist_name):
		super(QWidget,self).__init__()
		self.dist_types = ['beta','empty','gamma','normal','uniform']
		self.initialize(dist_name)

	def initialize(self,dist_name):
		self.hbox = QHBoxLayout()
		
		distlabel = QLabel(dist_name)
		
		self.prior_type = QComboBox()
		[self.prior_type.addItem(i) for i in self.dist_types]
		
		self.param1 = QLineEdit()
		self.param2 = QLineEdit()
		
		# self.prior_type.activated[str].connect(self.callback_type)
		# self.param1.textChanged[str].connect(self.callback_param)
		# self.param2.textChanged[str].connect(self.callback_param)
		
		self.hbox.addStretch(1)
		self.hbox.addWidget(distlabel)
		self.hbox.addWidget(self.prior_type)
		self.hbox.addWidget(self.param1)
		self.hbox.addWidget(self.param2)

		# self.button = QPushButton('test')
		# self.button.clicked.connect(self.callback_test)
		# self.hbox.addWidget(self.button)
		
		self.setLayout(self.hbox)
		self.show()
		
	# def callback_type(self,string):
	# 	if str(self.prior_type.currentText()) == 'empty':
	# 		[i.setDisabled(1) for i in [self.param1,self.param2]]
	# 	else:
	# 		[i.setEnabled(1) for i in [self.param1,self.param2]]
	#
	# def callback_param(self,string):
	# 	print str(self.param1.text())
	# 	print str(self.param2.text())
	# def callback_test(self):
	# 	print str(self.prior_type.currentText()), str(self.param1.text()), str(self.param2.text())
	
	
class priors(QWidget):
	def __init__(self):
		super(QWidget,self).__init__()
		self.priors_dists = [u'ϵ_1',u'ϵ_2',u'σ',u'k_1',u'k_2']
		self.initialize()

	def initialize(self):
		self.setWindowTitle('Set Priors')
		
		vbox = QVBoxLayout()
		hbox = QHBoxLayout()
		
		bcheck = QPushButton("Check")
		bview = QPushButton("View")
		bset = QPushButton("Set")
		bcheck.clicked.connect(self.check_dist)
		bview.clicked.connect(self.view_dist)
		bset.clicked.connect(self.set_dist)
		
		hbox.addStretch(1)
		[hbox.addWidget(i) for i in [bcheck,bset,bview]]
		
		vbox.addLayout(hbox)
		
		self.dists = [distribution(self.priors_dists[i]) for i in range(5)]
		[vbox.addWidget(self.dists[i]) for i in range(5)]
		vbox.addStretch(1)
		
		self.setLayout(vbox)
		self.show()
	
	def check_dist(self):
		pass
	def view_dist(self):
		pass
	def set_dist(self):
		pass
	
	def callback(self,*args):
		print self.width(), self.height()


if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = priors()
	sys.exit(app.exec_())