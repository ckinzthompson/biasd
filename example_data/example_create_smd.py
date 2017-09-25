''' This script loads the example data, and then creates an HDF5 SMD data file to containing this data. Future analysis performed with BIADS can also be saved into this file'''

## Imports
import numpy as np
import biasd as b

## Create a new SMD file
filename = './example_dataset.hdf5'
dataset = b.smd.new(filename,force=True)

## Add a custom attribute to describe the dataset
dataset.attrs.create('Description',
'''
This is the description of the dataset. blah blah blah.
'''
)

## Load example trajectories (N,T)
example_data = b.smd.loadtxt('example_data.dat')
n_molecules, n_datapoints = example_data.shape

## These signal versus time trajectories were simulated to be like smFRET data.
## The simulation parameters were:
tau = 0.1 # seconds
e1 = 0.1 # E_{FRET}
e2 = 0.9 # E_{FRET}
sigma = 0.05 #E_{FRET}
k1 = 3. # s^{-1}
k2 = 8. # s^{-1}

truth = np.array((e1,e2,sigma,k1,k2))

## Create a vector with the time of each datapoint
time = np.arange(n_datapoints) * tau

## Add the trajectories to the SMD file automatically
b.smd.add.trajectories(dataset, time, example_data, x_label='time', y_label='E_{FRET}')

## Add some metadata about the simulation to each trajectory
for i in range(n_molecules):

	# Select the group of interest
	trajectory = dataset['trajectory ' + str(i)]

	# Add an attribute called tau to the data group.
	# This group contains the time and signal vectors.
	trajectory['data'].attrs['tau'] = tau

	# Add a new group called simulation in the data group
	simulation = trajectory['data'].create_group('simulation')

	# Add relevant simulation paramters
	simulation.attrs['tau'] = tau
	simulation.attrs['e1'] = e1
	simulation.attrs['e2'] = e2
	simulation.attrs['sigma'] = sigma
	simulation.attrs['k1'] = k1
	simulation.attrs['k2'] = k2

	# Add an array of simulation parameters for easy access
	simulation.attrs['truth'] = truth

## Save the changes, and close the HDF5 file
b.smd.save(dataset)
