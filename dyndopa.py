import numpy as np
import json
import matplotlib
#matplotlib.use('Agg')
import pylab as pl
import pprint as pp
import sys
# Plot the different figures for the merged spikes and voltages recordings.
# This file, as the MergeSpikefiles.py should be one level up than Test/..., the output of a simulation.


path = sys.argv[1]+'/'

fparam = path+'Test/Parameters/simulation_parameters.json'
f = open(fparam, 'r')
params = json.load(f)

si = 11 
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

#pp.pprint(params)
recorder_type = 'spikes'
cell = 'rew'

outcome = np.loadtxt(path+params['rewards_multi_fn']+'_0')
data = np.loadtxt(path+params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )

color = ['b','g', 'r', 'c', 'm', 'y', 'k']
color = ['b','y', 'k', 'g', 'r', 'm', 'c']
z = 0
cl = color[z%len(color)]
cl = 'k'
size = 5.

print 'SPIKES'
ymax = 0. 
fig = pl.figure(123,(8,5))
ax = fig.add_subplot(211)
spread = 2
x = len(outcome) * spread 
tempd =np.sort(data[:,1])[params['t_init']:] 

#distrib = np.histogram(tempd, x)[0]
#distrib = np.histogram(data[:,1], x)[0]

#distrib = distrib[0] / float(np.sum(distrib[0]))
#distrib = distrib - np.mean(distrib)


#adapt = np.max(distrib)


#y=np.mean(distrib)
#pl.title('dopamine dynamic')
if len(sys.argv)==4:
    start = int(sys.argv[2])
    end= int(sys.argv[3])
    data[:,0]-=min(data[:,0])
    gids = []
    ttime = []
    for i in xrange(len(data[:,0])):
        if data[i,1]>start and data[i,1]<end:
            gids.append(data[i,0])
            ttime.append(data[i,1])
    ax.hist(ttime, len(ttime)/100.,  facecolor='black')
#    ax.set_xlim([start,end])
else:
    ax.hist(tempd, len(tempd)/100.,  facecolor='black')
    ax.set_xlim([0,params['t_sim']])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
pl.xticks(pl.xticks()[0],[str(int(a/1000.)) for a in pl.xticks()[0]])
#pl.xticks([])
pl.yticks(pl.yticks()[0],[str(int(a/10.)) for a in pl.yticks()[0]])
#ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
pl.ylabel('Firing rate [Hz]')

bx = fig.add_subplot(212)
print cell
data = np.loadtxt(path+params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )
if len(data)<2:
    print 'no data in ', cell
else:
    if len(sys.argv)==4:
        bx.scatter(ttime, gids, c=cl, s=size,label=cell, marker="|")
        bx.set_xlim([start,end])
        
    else:
        bx.scatter(data[:,1], data[:,0], c=cl, s=size,label=cell, marker="|")
        bx.set_xlim([0,params['t_sim']])
bx.set_ylim([0., max(gids)])
bx.spines['top'].set_visible(False)
bx.spines['right'].set_visible(False)
bx.spines['left'].set_visible(False)
bx.get_xaxis().tick_bottom()
bx.get_yaxis().tick_left()
bx.tick_params(axis='x', direction='out')
bx.tick_params(axis='y', length=0)
#bx.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
bx.set_axisbelow(True)
pl.xlabel('Time [s]')
pl.ylabel('Dopaminergic neuron ID')
pl.xticks(pl.xticks()[0],[str(int(a/1000.)) for a in pl.xticks()[0]])
pl.subplots_adjust(left = .04, bottom=.04, right=.97, top=.97)
#figman = pl.get_current_fig_manager()
#figman.frame.Maximize(True)
if len(sys.argv)==4:
    pl.savefig('zoom.pdf', bbox_inches='tight', dpi=1000)
    pl.savefig('zoom.png', bbox_inches='tight', dpi=1000)
    pl.savefig('zoom.tiff', bbox_inches='tight', dpi=1000)

pl.show()
