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
		
		self.prior_type.activated[str].connect(self.callback_type) 
		self.param1.textChanged[str].connect(self.callback_param) 
		self.param2.textChanged[str].connect(self.callback_param) 
		
		self.hbox.addWidget(distlabel)
		self.hbox.addWidget(self.prior_type)
		self.hbox.addWidget(self.param1)
		self.hbox.addWidget(self.param2)

		# self.button = QPushButton('test')
		# self.button.clicked.connect(self.callback_test)
		# self.hbox.addWidget(self.button)
		
		self.setLayout(self.hbox)
		self.show()
		
	def callback_type(self,string):
		if str(self.prior_type.currentText()) == 'empty':
			[i.setDisabled(1) for i in [self.param1,self.param2]]
		else:
			[i.setEnabled(1) for i in [self.param1,self.param2]]

	def callback_param(self,string):
		print str(self.param1.text())
		print str(self.param2.text())
	def callback_test(self):
		print str(self.prior_type.currentText()), str(self.param1.text()), str(self.param2.text())
	
	
class priors(QWidget):
	def __init__(self):
		super(QWidget,self).__init__()
		self.initialize()

	def initialize(self):
		
		self.setWindowTitle('HDF5 SMD Viewer')
		self.setMinimumSize(800,0)
		
		self.viewer = QColumnView()#QTreeView()#QListView()

		b = QPushButton("Select Dataset")
		b.clicked.connect(self.select_dataset)
		bget = QPushButton("Print Path")
		bget.clicked.connect(self.get_selection)
		bprint = QPushButton("Print Value")
		bprint.clicked.connect(self.print_selection)
		hbox = QHBoxLayout()

		hbox.addWidget(bget)
		hbox.addWidget(bprint)
		hbox.addStretch(1)
		if not self.parentWidget() is None:
		# if 1:
			hbox.addWidget(b)
			
		vbox = QVBoxLayout()
		vbox.addWidget(self.viewer)
		vbox.addLayout(hbox)
		
		self.setLayout(vbox)
		
		self.new_model(filename)
		if not self.model is None:
			self.viewer.setModel(self.model)
			self.show()
	
	def print_selection(self):
		try:
			print self.viewer.selectedIndexes()[0].data()
		except:
			pass
			
	def get_current_path(self):
		try:
			s = self.viewer.selectedIndexes()
			path = []
			if len(s) > 0:
				p = s[0]
				while True:
					if p.data() is None:
						break
					else:
						path.insert(0,p.data())
						p = p.parent()
			child = s[0].child(0,0).data()
			return path,child
		except:
			return [],None
	
	def get_selection(self,only_dataset=False):
		path,child = self.get_current_path()

		if path.count('attrs') > 0:
			path = path[:path.index('attrs')]
			child = ""
		
		if len(path) > 1:
			if child is None:
				child = path[-1]
				path = path[:-1]
			
			location = ''.join([pp+'/' for pp in path[1:]])[:-1]
			
			if only_dataset:
				if child.startswith('Dataset - shape: '):
					return location
			else:
				print "'" + location + "'"
	
	def select_dataset(self):
		location = self.get_selection(only_dataset=True)
		if not location is None:
			if self.parentWidget() is None:
				print "Sending: '"+location+"'"
			else:
				self.parentWidget().get_selected_dataset(location)
	
	def new_model(self,filename):
		def add_group(group,si):
			attrs = group.attrs.items()
			attr_si = QStandardItem('attrs')
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
	app = QApplication(sys.argv)
	
	dname = r'k_{1}'
	w = distribution(dname)
	sys.exit(app.exec_())