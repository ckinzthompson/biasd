"""
Practical Single-molecule Dataset Usage
	Adapted from the github smddata/smd-python script, so that now it's useful...
	CKT - 2016/06/01
---------------------------------------

Example:
import numpy as np
d = np.zeros((10,3,1000))
d[:,0] = np.random.normal(loc=300.,scale=10.,size=(10,1000))
d[:,1] = np.random.normal(loc=700.,scale=20.,size=(10,1000))
d[:,2] = d[:,1]/(d[:,0]+d[:,1])

s = smd.new(np.arange(1000)*.1,d,["Cy3","Cy5","FRET"])
smd.save("test.smd",s)
q = smd.load("test.smd")
"""

import numpy as _np

class _object_from_dict(object):
	""" Recursively turn a dict into an object"""
	def __init__(self, data):
		for key,value in data.items():
			if isinstance(value, (list, tuple)):
				setattr(self, key, [_object_from_dict(x) if isinstance(x, dict) else x for x in value])
				
			elif isinstance(value,dict):
				setattr(self, key, _object_from_dict(value))
			else:
				setattr(self, key, value)

def _dict_from_object(input_obj):
	""" Recursively turn an object into an dict"""
	out_dict = input_obj.__dict__
	for key,value in out_dict.items():
		if isinstance(value,(list,tuple)):
			out_dict[key] = [_dict_from_object(x) if isinstance(x,_object_from_dict) else x for x in value]
		elif isinstance(value,_object_from_dict):
			out_dict[key] = _dict_from_object(value)
	return out_dict

def _json_to_python(smd):
	""" Turn an SMD in JSON dict format to a python object """
	from copy import deepcopy
	out = _object_from_dict(deepcopy(smd))
	out.attr.SMD_format = 'Object'
	return out

def _python_to_json(smd):
	""" Turn an SMD in python object format to a JSON dict """
	from copy import deepcopy
	smd.attr.SMD_format = 'JSON'
	out = _dict_from_object(deepcopy(smd))
	return out

def test_smd(n_points=1000,n_traces=10):
	"""
	Returns a test, FRET dataset in SMD format
	"""
	from numpy import array,arange,random
	centers = array((150.,850.,0.))[None,:,None]
	t = arange(n_points,dtype='f')
	d = random.normal(scale = 10, size=(n_traces,3,n_points)) + centers
	d[:,2] = d[:,1]/(d[:,0]+d[:,1])
	smd = new(t,d,['Cy3','Cy5','FRET'])
	
	pb = random.randint(n_points/10,n_points,size=n_traces)
	for i in range(n_traces):
		smd.data[i].values.Cy3[pb[i]:] -= centers[0,0]
		smd.data[i].values.Cy5[pb[i]:] -= centers[0,1]
		smd.data[i].values.FRET[pb[i]:] = 0.
		smd.data[i].attr.pb_time = pb[i]
	return smd
	

def new(time,data,channel_names=None):
	"""
	time is T (number of datapoints)
	data is NxDxT because that's consistent with original SMD...
		where: 
			N: number traces
			D: data dimensionality i.e. number of channels
			T: number of time points
	channel_names is a list of strings, e.g. ["Cy3","Cy5"]
	"""
	if data.ndim == 2:
		data = data[:,None,:]
	if channel_names is None:
		channel_names = ["C"+str(i) for i in range(data.shape[1])]
	elif isinstance(channel_names,str):
		channel_names = [channel_names]
		
	types = dict([('index',time.dtype.name),('values',dict(zip(channel_names,[data.dtype.name for _ in range(data.shape[1])])))])
	
	import json
	import cPickle as pickle
	import numpy as np
	import hashlib

	# Hashes are tricky b/c not necessarily c-contiguous...
	smd = dict([('id', hashlib.md5(data.copy()).hexdigest() ),('attr',{'n_traces':data.shape[0]}),('types',types),('data',[])])
	for i in range(data.shape[0]):
		DataTypes = dict.items(types['values'])
		TempDict = types['values'].keys()
		TempDict = dict.fromkeys(TempDict)
		for j in range(len(types['values'])):
			TempDict[DataTypes[j][0]] = np.array(np.hstack(data[i,j]),dtype=DataTypes[j][1])
		index = np.array(time,dtype=types['index'])
		TempData = dict([('id',hashlib.md5(data[i].copy()).hexdigest()),('index',np.hstack(index)),('values',TempDict),('attr',{"trace_id":i})])
		smd['data'].append(TempData)
	
	smd = _json_to_python(smd)
	return smd

def _conversion_default(smd):
	"""
	Converts some SMD Things
	Default: Lists to Numpy
	"""
	from numpy import array
	for i in range(len(smd['data'])):
		s = smd['data'][i]
		# Time Vector
		s['index'] = array(s['index'])
		for key,value in s['values'].items():
			# Data Vectors
			s['values'][key] = array(value)
	return smd

def load(filename,JSON=False,conversion=_conversion_default):
	"""
	Give filename
	default returns SMD in Python Object form
	"""
	import json
	json_data = open(filename).read()
	smd = json.loads(json_data)
	
	if hasattr(conversion, '__call__'):
		smd = conversion(smd)
	
	if not JSON:
		smd = _json_to_python(smd)

	return smd

def save(filename,smd):
	'''
	Save your SMD file in filename... will save as JSON.... so all numpy things are turned into lists... yay.
	
	'''
	import json
	import numpy
	
	if filename.endswith('.smd'):
		filename = filename[:-4]
	if isinstance(smd,_object_from_dict):
		smd = _python_to_json(smd)
		
	class NumpyAwareJSONEncoder(json.JSONEncoder):
		def default(self, obj):
			if isinstance(obj, numpy.ndarray) and obj.ndim == 1:
				return obj.tolist()
			elif isinstance(obj,dict):
				for k,v in obj.items():
					if isinstance(v,numpy.ndarray):
						obj[k] = v.tolist()
			return json.JSONEncoder.default(self, obj)
			
	
	j=json.dumps(smd,cls=NumpyAwareJSONEncoder, sort_keys=True, indent=4, separators=(',', ': '))
	
	f=open((filename+'.smd'),"w")
	f.write(j)
	f.close()
	return j


class smd_io_error(Exception):
	def __init__(self):
		Exception.__init__(self,"SMD I/O error") 