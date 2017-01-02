'''
GUI written in QT5 to explore HDF5 SMD Files
'''
from h5py import File
from PyQt5.QtWidgets import QApplication, QColumnView, QWidget, QVBoxLayout
from PyQt5.QtGui import QStandardItemModel, QStandardItem

def apply_dark_theme(qApp):
	# From https://gist.github.com/lschmierer/443b8e21ad93e2a2d7eb
	from PyQt5.QtGui import QPalette,QColor
	from PyQt5.QtCore import Qt

	dark_palette = QPalette()

	dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
	dark_palette.setColor(QPalette.WindowText, Qt.white)
	dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
	dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
	dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
	dark_palette.setColor(QPalette.ToolTipText, Qt.white)
	dark_palette.setColor(QPalette.Text, Qt.white)
	dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
	dark_palette.setColor(QPalette.ButtonText, Qt.white)
	dark_palette.setColor(QPalette.BrightText, Qt.red)
	dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
	dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
	dark_palette.setColor(QPalette.HighlightedText, Qt.black)

	qApp.setPalette(dark_palette)

	qApp.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

class smd_view(QWidget):	
	def __init__(self,filename=None):
		super(QWidget,self).__init__()
		self.initialize(filename)

	def initialize(self,filename):
		
		self.setWindowTitle('HDF5 SMD Viewer')
		self.setMinimumSize(800,0)
		
		self.viewer = QColumnView()#QTreeView()#QListView()
		vbox = QVBoxLayout()
		vbox.addWidget(self.viewer)
		vbox.addStretch(1)
		self.setLayout(vbox)
		
		self.new_model(filename)
		if not self.model is None:
			self.viewer.setModel(self.model)
			self.show()
		
	def new_model(self,filename):
		def add_group(group,si):
			attrs = group.attrs.items()
			attr_si = QStandardItem('Attributes')
			for i in range(len(attrs)):
				attr_child = QStandardItem(attrs[i][0])
				attr_child.setChild(0,QStandardItem(str(attrs[i][1])))
				attr_si.setChild(i,attr_child)
			si.setChild(0,attr_si)
	
			groups  = group.items()
			# Add 1 b/c attributes is 0
			for i in range(len(groups)):
				newthing = QStandardItem(groups[i][0])
				try:
					add_group(groups[i][1],newthing)
					si.setChild(i+1,newthing)
				except:
					try:
						newthing.setChild(0,QStandardItem("Dataset - shape: "+str(groups[i][1].value.shape)))
						si.setChild(i+1,newthing)
					except:
						print groups[i][0]
			
		self.model = QStandardItemModel(self.viewer)
		try:
			f = File(filename,'r')
			dataset = QStandardItem(f.filename)
			add_group(f,dataset)
			f.close()
			self.model.appendRow(dataset)
			
		except:
			print "couldn't load file"
			

if __name__ == '__main__':
	import sys
	fn = '/Users/colin/Desktop/20161220 biasd_release/biasd/example_dataset.hdf5'
	app = QApplication(sys.argv)
	apply_dark_theme(app)
	w = smd_view(fn)
	sys.exit(app.exec_())