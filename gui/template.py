# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to perform ...
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow, QShortcut
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import sys
biasd_path = '../'
sys.path.append(biasd_path)
import biasd as b

class temp(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		self.setWindowTitle('')
		self.show()
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
	
	def make_shortcut(self,key,fxn):
		qs = QShortcut(self)
		qs.setKey(key)
		qs.activated.connect(fxn)
			
	def init_shortcuts(self):
		#self.make_shortcut("o",self.dothis)
			

class ui_temp(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = temp(self)
		self.setCentralWidget(self.ui)
		self.show()
	
	def closeEvent(self,event):
		self.parent().activateWindow()
		self.parent().raise_()
		self.parent().setFocus()

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_temp()
	sys.exit(app.exec_())
