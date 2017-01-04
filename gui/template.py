# -*- coding: utf-8 -*-®
'''
GUI written in QT5 to setup preferences
'''
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel, QLineEdit, QMessageBox, QMainWindow
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

import sys
biasd_path = '/Users/colin/Desktop/20161220 biasd_release/biasd'
sys.path.append(biasd_path)
import biasd as b


class prefs(QWidget):
	def __init__(self,parent):
		super(QWidget,self).__init__(parent=parent)
		self.initialize()

	def initialize(self):
		self.setWindowTitle('Set Preferences')
		# self.setGeometry(200,200,500,300)
		self.show()
		
	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Escape:
			self.parent().close()
			

class ui_preferences(QMainWindow):
	def __init__(self,parent=None):
		super(QMainWindow,self).__init__(parent)
		self.ui = prefs(self)
		self.setCentralWidget(self.ui)
		self.setGeometry(100,100,400,300)
		self.show()
	
	def closeEvent(self,event):
		pass

		
if __name__ == '__main__':
	import sys
	app = QApplication(sys.argv)
	w = ui_preferences()
	sys.exit(app.exec_())