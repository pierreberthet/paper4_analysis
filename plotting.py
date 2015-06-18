import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import pylab as pl

# Plot the different figures for the merged spikes and voltages recordings.
# This file, as the MergeSpikefiles.py should be one level up than Test/..., the output of a simulation.

fparam = 'Test/Parameters/simulation_parameters.json'
f = open(fparam, 'r')
params = json.load(f)

params['figures_folder'] = "%sFigures/" % params['folder_name']

 

i_f = 1

pl.figure(i_f)

cell = 'states'
recorder_type = 'spikes'

for nstate in range(params['n_states']):
    #    print nstate
    data = np.loadtxt(params['spiketimes_folder']+'/'+str(nstate)+cell+'_merged_'+recorder_type+'.dat' )
    pl.scatter(data[:,1], data[:,0])
pl.title(cell)
name= 'fig'+str(i_f)+'_'+cell+'.pdf'
pl.savefig(params['figures_folder']+name)
pl.show()

i_f += 1

cell = 'rp'
pl.figure(i_f)
for ni in range(params['n_states']*params['n_actions']):
    #    print nstate
    data = np.loadtxt(params['spiketimes_folder']+str(ni)+cell+'_merged_'+recorder_type+'.dat' )
    pl.scatter(data[:,1], data[:,0])
pl.title(cell)
name= 'fig'+str(i_f)+'_'+cell+'.pdf'
pl.savefig(params['figures_folder']+name)


i_f += 1

cell = 'rew'
pl.figure(i_f)
data = np.loadtxt(params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )
pl.scatter(data[:,1], data[:,0])
pl.title(cell)
name= 'fig'+str(i_f)+'_'+cell+'.pdf'
pl.savefig(params['figures_folder']+name)

i_f += 1


pl.show()

cell_types = ['d1', 'd2', 'actions','efference']
cell_types_volt = ['d1', 'd2', 'actions']

# SPIKES
for cell in cell_types:
    pl.figure(i_f)
    for naction in range(params['n_actions']):
        data = np.loadtxt(params['spiketimes_folder']+str(naction)+cell+'_merged_'+recorder_type+'.dat' )
        pl.scatter(data[:,1], data[:,0])
        
    pl.title(cell)
    name= 'fig'+str(i_f)+'_'+cell+'.pdf'
    pl.savefig(params['figures_folder']+name)
    i_f += 1

pl.show()
'''
# VOLTAGES
for cell in cell_types_volt:
    for n in params['n_actions']:
'''


