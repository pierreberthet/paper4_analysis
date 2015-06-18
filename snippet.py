import numpy as np
import json
import matplotlib
#matplotlib.use('Agg')
import pylab as pl
import pprint as pp
import sys
# Plot the different figures for the merged spikes and voltages recordings.
# This file, as the MergeSpikefiles.py should be one level up than Test/..., the output of a simulation.
path =sys.argv[1]+'/' 
fparam = path+'Test/Parameters/simulation_parameters.json'
f = open(fparam, 'r')
params = json.load(f)

#pl.clf()
#pp.pprint(params)
if len(sys.argv)==4:
    start = int(sys.argv[2])
    end= int(sys.argv[3])

si = 25 
parms = {
    'axes.labelsize': si,
    'text.fontsize': si,
    'legend.fontsize': si,
    'xtick.labelsize': si,
    'ytick.labelsize': si,
    'text.usetex': False
    #'figure.figsize': [6., 7.]
}
pl.rcParams.update(parms)

params['figures_folder'] = "%sFigures" % params['folder_name']
#color = ['b','g', 'r', 'c', 'm', 'y', 'k']
color = ['b','g', 'r', 'm', 'c', 'y', 'k']
#color = ['b','y', 'k', 'g', 'r', 'm', 'c']
z = 0
cl = color[z%len(color)]

xa = -(params['t_sim']/10)
size = 5.

print 'SPIKES'
ymax = 0. 
#fig = pl.figure(1,(50,11.5))
fig = pl.figure(1,(21,11.5))
ax = fig.add_subplot(111)


cell = 'states'
recorder_type = 'spikes'

mean = 0
ymax = 0

print cell
mini = 391
for nstate in range(params['n_states']):
    #    print nstate
    data = np.loadtxt(path+params['spiketimes_folder']+'/'+str(nstate)+cell+'_merged_'+recorder_type+'.dat' )
    #data[:,0]-=min(data[:,0])
    gids = []
    ttime = []
    for i in xrange(len(data[:,0])):
        if data[i,1]>start and data[i,1]<end:
            gids.append(data[i,0])
            ttime.append(data[i,1])
    gids = [g-mini for g in gids]
    ymax = max(ymax , max(gids))
        
    #mean += (np.min(data[:,0])+np.max(data[:,0]))/2
#    if np.max(data[:,0]) > ymax:
#        ymax = np.max(data[:,0])
    ax.scatter(ttime, gids, c=cl, s=size, marker="|")
mean = mean/params['n_states']
#ax.text(xa, mean, cell, color=cl)

z += 1
cl = color[z%len(color)]
#cell = 'rp'
#print cell
#for ni in range(params['n_states']*params['n_actions']):
#    #    print nstate
#    data = np.loadtxt(params['spiketimes_folder']+str(ni)+cell+'_merged_'+recorder_type+'.dat' )
#    if len(data)<=2:
#       print 'no data in ', cell, ni 
#    else:
#        mean += (np.min(data[:,0])+np.max(data[:,0]))/2
#        if np.max(data[:,0]) > ymax:
#            ymax = np.max(data[:,0])
#        pl.scatter(data[:,1], data[:,0], c=cl, s=size, marker="|")
#
#mean = mean/(params['n_states']*params['n_actions'])
#pl.text(xa, mean, cell, color=cl)
#mean = 0
#z += 1
#cl = color[z%len(color)]
#
#
#cell = 'rew'
#print cell
#data = np.loadtxt(params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )
#if len(data)<2:
#    print 'no data in ', cell
#else:
#    pl.scatter(data[:,1], data[:,0], c=cl, s=size,label=cell, marker="|")
#    mean += (np.min(data[:,0])+np.max(data[:,0]))/2
#    if np.max(data[:,0]) > ymax:
#        ymax = np.max(data[:,0])
#    pl.text(xa, mean, cell, color=cl)
#    mean = 0
#    z += 1
#    cl = color[z%len(color)]



#cell_types = ['d1', 'd2', 'actions','efference', 'brainstem']
#cell_types = ['d1', 'd2', 'actions','efference']
cell_types = ['d1', 'd2']

# SPIKES
for cell in cell_types:
    print cell
    for naction in range(params['n_actions']):
        data = np.loadtxt(path+params['spiketimes_folder']+str(naction)+cell+'_merged_'+recorder_type+'.dat' )
        gids = []
        ttime = []
        for i in xrange(len(data[:,0])):
            if data[i,1]>start and data[i,1]<end:
                gids.append(data[i,0])
                ttime.append(data[i,1])
        gids = [g-mini for g in gids]
        ax.scatter(ttime, gids, c=cl, s=size, marker="|")
        mean += (np.min(data[:,0])+np.max(data[:,0]))/2
       # if np.max(data[:,0]) > ymax:
       #     ymax = np.max(data[:,0])
    mean = mean/params['n_actions']
    #pl.text(xa,mean, cell, color=cl)
    z += 1
    cl = color[z%len(color)]
    mean = 0
cell_types=['actions']
mini=245
for cell in cell_types:
    print cell
    for naction in range(params['n_actions']):
        data = np.loadtxt(path+params['spiketimes_folder']+str(naction)+cell+'_merged_'+recorder_type+'.dat' )
        gids = []
        ttime = []
        for i in xrange(len(data[:,0])):
            if data[i,1]>start and data[i,1]<end:
                gids.append(data[i,0])
                ttime.append(data[i,1])
        gids = [g-mini for g in gids]
        ax.scatter(ttime, gids, c=cl, s=size, marker="|")
        mean += (np.min(data[:,0])+np.max(data[:,0]))/2
       # if np.max(data[:,0]) > ymax:
       #     ymax = np.max(data[:,0])
    mean = mean/params['n_actions']
    #pl.text(xa,mean, cell, color=cl)
    z += 1
    cl = color[z%len(color)]
    mean = 0
ax.set_ylim([0, ymax])
ax.set_xlim([start, end])
lines= np.arange(params['t_init'],params['t_sim'], params['t_iteration'])
ax.vlines(lines, [0], ymax,  color='0.55', linestyles='dashed')
#pl.title(str(params['n_states'])+' states '+str(params['n_actions'])+' actions '+str(params['n_blocks']*params['block_len'])+ ' trials' )
ax.set_ylabel("Neuron ID")
#ax.set_xlabel("Time [s]")
#pl.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
#ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
pl.xticks(pl.xticks()[0],[str(int(a/1000.)) for a in pl.xticks()[0]])
#pl.xticks([])
pl.tight_layout()
pl.subplots_adjust(left = .06, bottom=.06, right=.98, top=.99)
pl.savefig('snippet.png', bbox_inches='tight', dpi=800)
pl.savefig('snippet.pdf', bbox_inches='tight', dpi=400)
pl.savefig('snippet.tiff', bbox_inches='tight', dpi=400)
pl.show()

pl.clf()
pl.close()
