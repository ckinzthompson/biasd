''' This script loads the example data, sets some priors, and then uses the Markov chain Monte Carlo (MCMC) technique to sample the posterior.'''

## Imports
import matplotlib.pyplot as plt
import numpy as np
import biasd as b


#### Setup the analysis
## Load the SMD example dataset
filename = './example_dataset.hdf5'
dataset = b.smd.load(filename)

## Select the data from the first trajectory
trace = dataset['trajectory 0']
time = trace['data/time'].value
fret = trace['data/E_{FRET}'].value

## Parse meta-data to load time resolution
tau = trace['data'].attrs['tau']

## Get the simulation ground truth values 
truth = trace['data/simulation'].attrs['truth']

## Close the dataset
dataset.close()


#### Perform a Calculation
## Make the prior distribution
## set means to ground truths: (.1, .9, .05, 3., 8.)
e1 = b.distributions.normal(0.1, 0.2)
e2 = b.distributions.normal(0.9, 0.2)
sigma = b.distributions.gamma(1., 1./0.05)
k1 = b.distributions.gamma(1., 1./3.)
k2 = b.distributions.gamma(1., 1./8.)
priors = b.distributions.parameter_collection(e1,e2,sigma,k1,k2)

## Setup the MCMC sampler to use 100 walkers and 4 CPUs
nwalkers = 100
ncpus = 4
sampler, initial_positions = b.mcmc.setup(fret, priors, tau, nwalkers, threads = ncpus)

## Burn-in 100 steps and then remove them form the sampler,
## but keep the final positions
sampler, burned_positions = b.mcmc.burn_in(sampler,initial_positions,nsteps=100)

## Run 100 steps starting at the burned-in positions. Timing data will provide an idea of how long each step takes
sampler = b.mcmc.run(sampler,burned_positions,nsteps=100,timer=True)

## Continue on from step 100 for another 900 steps. Don't display timing.
sampler = b.mcmc.continue_run(sampler,900,timer=False)

## Get uncorrelated samples from the chain by skipping samples according to the autocorrelation time of the variable with the largest autocorrelation time
uncorrelated_samples = b.mcmc.get_samples(sampler,uncorrelated=True)

## Make a corner plot of these uncorrelated samples
fig = b.mcmc.plot_corner(uncorrelated_samples)
fig.savefig('example_mcmc_corner.png')


#### Save the analysis
## Create a new group to hold the analysis in 'trajectory 0'
dataset = b.smd.load(filename)
trace = dataset['trajectory 0']
mcmc_analysis = trace.create_group("MCMC analysis 20170106")

## Add the priors
b.smd.add.parameter_collection(mcmc_analysis,priors,label='priors')

## Extract the relevant information from the sampler, and save this in the SMD file.
result = b.mcmc.mcmc_result(sampler)
b.smd.add.mcmc(mcmc_analysis,result,label='MCMC posterior samples')

## Save and close the dataset
b.smd.save(dataset)
