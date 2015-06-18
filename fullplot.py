import numpy as np
import json
import matplotlib
#matplotlib.use('Agg')
import pylab as pl
import pprint as pp
# Plot the different figures for the merged spikes and voltages recordings.
# This file, as the MergeSpikefiles.py should be one level up than Test/..., the output of a simulation.

fparam = 'Test/Parameters/simulation_parameters.json'
f = open(fparam, 'r')
params = json.load(f)

pp.pprint(params)


params['figures_folder'] = "%sFigures" % params['folder_name']
color = ['b','g', 'r', 'c', 'm', 'y', 'k']
color = ['b','y', 'k', 'g', 'r', 'm', 'c']
z = 0
cl = color[z%len(color)]

xa = -(params['t_sim']/10)
size = 5.

print 'SPIKES'
ymax = 0. 
pl.figure(666)

cell = 'states'
recorder_type = 'spikes'

mean = 0

print cell

for nstate in range(params['n_states']):
    #    print nstate
    data = np.loadtxt(params['spiketimes_folder']+'/'+str(nstate)+cell+'_merged_'+recorder_type+'.dat' )
    mean += (np.min(data[:,0])+np.max(data[:,0]))/2
    if np.max(data[:,0]) > ymax:
        ymax = np.max(data[:,0])
    pl.scatter(data[:,1], data[:,0], c=cl, s=size, marker="|")
mean = mean/params['n_states']
pl.text(xa, mean, cell, color=cl)

z += 1
cl = color[z%len(color)]
cell = 'rp'
print cell
for ni in range(params['n_states']*params['n_actions']):
    #    print nstate
    data = np.loadtxt(params['spiketimes_folder']+str(ni)+cell+'_merged_'+recorder_type+'.dat' )
    if len(data)<=2:
       print 'no data in ', cell, ni 
    else:
        mean += (np.min(data[:,0])+np.max(data[:,0]))/2
        if np.max(data[:,0]) > ymax:
            ymax = np.max(data[:,0])
        pl.scatter(data[:,1], data[:,0], c=cl, s=size, marker="|")

mean = mean/(params['n_states']*params['n_actions'])
pl.text(xa, mean, cell, color=cl)
mean = 0
z += 1
cl = color[z%len(color)]


cell = 'rew'
print cell
data = np.loadtxt(params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )
if len(data)<2:
    print 'no data in ', cell
else:
    pl.scatter(data[:,1], data[:,0], c=cl, s=size,label=cell, marker="|")
    mean += (np.min(data[:,0])+np.max(data[:,0]))/2
    if np.max(data[:,0]) > ymax:
        ymax = np.max(data[:,0])
    pl.text(xa, mean, cell, color=cl)
    mean = 0
    z += 1
    cl = color[z%len(color)]



cell_types = ['d1', 'd2', 'actions','efference', 'brainstem']
#cell_types = ['d1', 'd2', 'actions','efference']

# SPIKES
for cell in cell_types:
    print cell
    for naction in range(params['n_actions']):
        data = np.loadtxt(params['spiketimes_folder']+str(naction)+cell+'_merged_'+recorder_type+'.dat' )
        if len(data)<2:
            print 'no data in', cell, naction
        else:
            if naction ==0:
                pl.scatter(data[:,1], data[:,0], c=cl, s=size, label=cell, marker="|")
            else:
                pl.scatter(data[:,1], data[:,0], c=cl, s=size, marker="|")
            mean += (np.min(data[:,0])+np.max(data[:,0]))/2
            if np.max(data[:,0]) > ymax:
                ymax = np.max(data[:,0])
    mean = mean/params['n_actions']
    pl.text(xa,mean, cell, color=cl)
    z += 1
    cl = color[z%len(color)]
    mean = 0
pl.ylim(0, ymax)
lines= np.arange(params['t_init'],params['t_sim'], params['t_iteration'])
pl.vlines(lines, [0], ymax,  color='0.55', linestyles='dashed')
pl.title(str(params['n_states'])+' states '+str(params['n_actions'])+' actions '+str(params['n_blocks']*params['block_len'])+ ' trials' )
pl.ylabel("cell GID")
pl.xlabel("time "+ r"$ms$")
pl.legend()
#pl.tight_layout()
pl.subplots_adjust(left = .04, bottom=.04, right=.97, top=.97)
pl.show()


