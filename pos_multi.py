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
#print fparam
f = open(fparam, 'r')
params = json.load(f)


si = 15 
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

params['multi_n']+=1

source_d1 =    path+ params['weights_d1_multi_fn']+'_'
source_d2 =    path+ params['weights_d2_multi_fn']+'_'
source_rew =   path+ params['rewards_multi_fn']+'_'
source_rp =    path+ params['weights_rp_multi_fn']+'_'
print 'init phase'
color = ['b','g', 'r', 'c', 'm', 'y', 'k']
z = 0
cl = color[z%len(color)]

lend1= len(np.loadtxt(source_d1+'0'))
lend2= len(np.loadtxt(source_d2+'0'))
lenrp= len(np.loadtxt(source_rp+'0'))
if not lend1 == lend2:
    print 'INCONSISTENCY D1 and D2 length (number of recordings)'
#params['multi_n'] = params['multi_n'] -1
wd1 = np.zeros((params['multi_n'], lend1, params['n_actions']))
wd1_m = np.zeros((lend1, params['n_actions']))
wd1_std = np.zeros((lend1, params['n_actions']))
wd2 = np.zeros((params['multi_n'], lend2,params['n_actions']))
wd2_m = np.zeros((lend2, params['n_actions']))
wd2_std = np.zeros((lend2, params['n_actions']))
wrp = np.zeros((params['multi_n'], lenrp,params['n_actions'] * params['n_states']))
wrp_m = np.zeros((lenrp, params['n_actions'] * params['n_states']))
wrp_std = np.zeros((lenrp, params['n_actions'] * params['n_states']))
rewards = np.zeros((params['multi_n'], params['n_iterations']))
rewards_m = np.zeros(params['n_iterations'])
rewards_std = np.zeros(params['n_iterations'])

for i in xrange(params['multi_n']):
    wd1[i] = np.loadtxt(source_d1+str(i))
    wd2[i] = np.loadtxt(source_d2+str(i))
    wrp[i] = np.loadtxt(source_rp+str(i))
    rewards[i] = np.loadtxt(source_rew+str(i))
    for x in xrange(lend1):
        for y in xrange(params['n_actions']):
            if wd1[i,x,y]<0.:
                wd1[i,x,y] = 0.
            if wd2[i,x,y]<0.:
                wd2[i,x,y] = 0.




for i in xrange(lend1):
    for j in xrange(params['n_actions']):
        wd1_m[i,j] = np.mean(wd1[:,i,j])
        wd1_std[i,j] = np.std(wd1[:,i,j])
        wd2_m[i,j] = np.mean(wd2[:,i,j])
        wd2_std[i,j] = np.std(wd2[:,i,j])
for i in xrange(lenrp):
    for j in xrange(params['n_actions']*params['n_states']):
        wrp_m[i,j] = np.mean(wrp[:,i,j])
        wrp_std[i,j] = np.std(wrp[:,i,j])

#wd1 = wd1 / params['multi_n']
#wd2 = wd2 / params['multi_n']
#rewards = rewards / params['multi_n']

mini = np.minimum(np.min(wd1_m - wd1_std), np.min(wd2_m - wd2_std)) - .1
maxi = np.maximum(np.max(wd1_m + wd1_std), np.max(wd2_m + wd2_std)) + .1
print 'd1 data' 

pl.figure(600)
ax = pl.subplot(211)
pl.title('D1')
down = np.zeros((lend1, params['n_actions']))
for i in xrange(lend1):
    for j in xrange(params['n_actions']):
        if (wd1_m[i,j]-wd1_std[i,j]>0.):
            down[i,j]= wd1_m[i,j]-wd1_std[i,j]
for i in xrange(params['n_actions']):
    pl.plot(wd1_m[:,i], c=cl)
    pl.fill_between(np.arange(lend1), wd1_m[:,i] + wd1_std[:,i], down[:,i], facecolor=cl,alpha =.5 )
    z+=1
    cl = color[z%len(color)]
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)

xmax= params['n_recordings']
mean = np.zeros(lend1)
for j in xrange(lend1):
    mean[j] = np.mean(wd1_m[j,:])
pl.plot(mean, 'k.')
pl.ylim([mini, maxi])
#pl.xlim([0., xmax])
pl.xticks(pl.xticks()[0],[str(int(a*.25)) for a in pl.xticks()[0]])
pl.ylabel('Average 'r'$W_{1j}$')
#pl.ylabel(r'$W_{ij}$')
#pl.vlines( np.arange(0,params['t_sim']/params['resolution'], params['t_sim']/(params['n_blocks']*params['resolution'])), [0], [1.01], color='0.55', linestyles='dashed' )
if params['n_blocks']>1:
    pl.vlines( np.arange(lend1/params['n_blocks'], lend1-1., lend1/params['n_blocks'] ), [mini], [maxi], color='0.55', linestyles='dashed' )

z=0
cl = color[z%len(color)]
print 'd2 data' 
ax =pl.subplot(212)
mean = np.zeros(lend2)
pl.title('D2')
down = np.zeros((lend1, params['n_actions']))
for i in xrange(lend2):
    for j in xrange(params['n_actions']):
        if (wd2_m[i,j]-wd2_std[i,j]>0.):
            down[i,j]= wd2_m[i,j]-wd2_std[i,j]
for i in xrange(params['n_actions']):
    pl.plot(wd2_m[:,i], c=cl)
    pl.fill_between(np.arange(lend2), wd2_m[:,i] + wd2_std[:,i], down[:,i], alpha =.5, facecolor=cl )
    z+=1
    cl = color[z%len(color)]
for j in xrange(lend2):
    mean[j] = np.mean(wd2_m[j,:])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
pl.plot(mean, 'k.')
pl.ylim([mini, maxi])
#pl.xlim([0., xmax])
#pl.xlim([0, maxi])
pl.ylabel('Average 'r'$W_{1j}$')
#pl.xlabel('time in '+str(params['resolution'])+r' $ms$')
pl.xlabel('Time [s]')
#pl.xticks(np.arange(0, params['n_iterations'], params['block_len']))
pl.xticks(pl.xticks()[0],[str(int(a*.25)) for a in pl.xticks()[0]])
#pl.vlines( np.arange(0,params['t_sim']/params['resolution'], params['t_sim']/(params['n_blocks']*params['resolution'])), [0], [1.01], color='0.55', linestyles='dashed' )
if params['n_blocks']>1:
    pl.vlines( np.arange(lend2/params['n_blocks'], lend2-1., lend2/params['n_blocks'] ), [mini], [maxi], color='0.55', linestyles='dashed' )

pl.savefig('d1d2weight.tiff', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight.pdf', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight.svg', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight.eps', bbox_inches='tight', dpi=600)
pl.figure(606)
ax = pl.subplot(211)
pl.title('D1')
z=0
cl = color[z%len(color)]
start = 5.*params['block_len']*params['t_iteration']/params['resolution']
mini = np.minimum(np.min(wd1_m[start:,:] - wd1_std[start:,:]), np.min(wd2_m[start:,:] - wd2_std[start:,:]))# - .1
maxi = np.maximum(np.max(wd1_m[start:,:] + wd1_std[start:,:]), np.max(wd2_m[start:,:] + wd2_std[start:,:]))# + .1
for i in xrange(params['n_actions']):
    pl.plot(wd1_m[start:,i], c=cl)
    pl.fill_between(np.arange(lend1-start), wd1_m[start:,i] + wd1_std[start:,i], wd1_m[start:,i] - wd1_std[start:,i], facecolor=cl,alpha =.5 )
    z+=1
    cl = color[z%len(color)]
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)

xmax= params['n_recordings']
mean = np.zeros(lend1)
for j in xrange(lend1):
    mean[j] = np.mean(wd1_m[j,:])
pl.plot(mean[start:], 'k.')
pl.ylim([mini, maxi])
#pl.xlim([0., xmax])
pl.xticks(pl.xticks()[0],[str(int((a+start)*.25)) for a in pl.xticks()[0]])
pl.ylabel('Average 'r'$W_{1j}$')
#pl.ylabel(r'$W_{ij}$')
#pl.vlines( np.arange(0,params['t_sim']/params['resolution'], params['t_sim']/(params['n_blocks']*params['resolution'])), [0], [1.01], color='0.55', linestyles='dashed' )
if params['n_blocks']>1:
    pl.vlines( np.arange((lend1-start)/(params['n_blocks']-5.), lend1-start-1., lend1/params['n_blocks'] ), [mini], [maxi], color='0.55', linestyles='dashed' )

z=0
cl = color[z%len(color)]
print 'd2 data' 
ax =pl.subplot(212)
mean = np.zeros(lend2)
pl.title('D2')
for i in xrange(params['n_actions']):
    pl.plot(wd2_m[start:,i], c=cl)
    pl.fill_between(np.arange(lend2-start), wd2_m[start:,i] + wd2_std[start:,i], wd2_m[start:,i] - wd2_std[start:,i], alpha =.5, facecolor=cl )
    z+=1
    cl = color[z%len(color)]
for j in xrange(lend2):
    mean[j] = np.mean(wd2_m[j,:])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
pl.plot(mean[start:], 'k.')
pl.ylim([mini, maxi])
#pl.xlim([0., xmax])
#pl.xlim([0, maxi])
pl.ylabel('Average 'r'$W_{1j}$')
#pl.xlabel('time in '+str(params['resolution'])+r' $ms$')
pl.xlabel('Time [s]')
#pl.xticks(np.arange(0, params['n_iterations'], params['block_len']))
pl.xticks(pl.xticks()[0],[str(int((start+a)*.25)) for a in pl.xticks()[0]])
#pl.vlines( np.arange(0,params['t_sim']/params['resolution'], params['t_sim']/(params['n_blocks']*params['resolution'])), [0], [1.01], color='0.55', linestyles='dashed' )
if params['n_blocks']>1:
    pl.vlines( np.arange((lend2-start)/(params['n_blocks']-5.), lend2-start-1., (lend2)/(params['n_blocks']) ), [mini], [maxi], color='0.55', linestyles='dashed' )

pl.savefig('d1d2weight_maxi.tiff', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight_maxi.pdf', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight_maxi.svg', bbox_inches='tight', dpi=600)
pl.savefig('d1d2weight_maxi.eps', bbox_inches='tight', dpi=600)

print 'rp data' 
fig = pl.figure(601)
ax=fig.add_subplot(111)
z=0
mean = np.zeros(lenrp)
cl = color[z%len(color)]
#pl.xticks(np.arange(0,params['t_sim'], params['resolution']))
#pl.xlabel('time in '+str(params['resolution'])+r' $ms$')
#pl.xlabel('time in '+str(params['resolution'])+r' $ms$')
pl.xlabel('Time [s]')
#pl.ylabel('average RP weights')
pl.ylabel('RP average 'r'$W_{ij}$')
for i in xrange(params['n_actions']*params['n_states']):
    ax.plot(wrp_m[:,i], c=cl)
    ax.fill_between(np.arange(lenrp), wrp_m[:,i] + wrp_std[:,i], wrp_m[:,i] - wrp_std[:,i], alpha =.5, facecolor=cl )
    z+=1
    cl = color[z%len(color)]
mini = np.min(wrp_m - wrp_std)
maxi = np.max(wrp_m + wrp_std)
if params['n_blocks']>1:
    ax.vlines( np.arange(lenrp/params['n_blocks'], lenrp-1., lenrp/params['n_blocks'] ), mini, maxi, color='0.55', linestyles='dashed' )
for j in xrange(lenrp):
    mean[j] = np.mean(wrp_m[j,:])
pl.ylim([mini, maxi])
#pl.xlim([0., xmax])
pl.xticks(pl.xticks()[0],[str(int(a*.25)) for a in pl.xticks()[0]])
ax.plot(mean, 'k.')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)

pl.savefig('rpweight.tiff', bbox_inches='tight', dpi=600)
pl.savefig('rpweight.pdf', bbox_inches='tight', dpi=600)
pl.savefig('rpweight.svg', bbox_inches='tight', dpi=600)
pl.savefig('rpweight.eps', bbox_inches='tight', dpi=600)

perf= np.zeros((params['multi_n'], params['n_iterations']))
for j in xrange(0, params['multi_n']):
    for i in xrange(1, params['n_iterations']):
        perf[j,i]= perf[j,i-1] + (rewards[j,i] - perf[j,i-1])*0.25
for m in xrange(params['n_iterations']):
    rewards_m[m] = np.mean(perf[:,m])
    rewards_std[m] = np.std(perf[:,m])
print 'average perf'
fig = pl.figure(602)
#pl.title('performance')
ax = fig.add_subplot(111)
pl.xlabel('Trials')
pl.ylabel('Average success ratio')
if params['n_blocks']>1:
    ax.vlines( np.arange(params['block_len'],params['n_blocks']*params['block_len']-1., params['block_len']), [0], [1.0], color='0.55', linestyles='dashed' )
ax.plot(rewards_m)

top = np.ones(len(rewards_m))
down = np.zeros(len(rewards_m))
for i in xrange(len(rewards_m)):
    if (rewards_m[i]+rewards_std[i]<1.):
        top[i]=rewards_m[i]+rewards_std[i]
    if (rewards_m[i]-rewards_std[i]>0.):
        down[i]=rewards_m[i]-rewards_std[i]

ax.fill_between(np.arange(params['n_iterations']), top,down, alpha=.2)
pl.ylim([0, 1.0])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_axisbelow(True)
pl.savefig('perf.tiff', bbox_inches='tight', dpi=600)
pl.savefig('perf.pdf', bbox_inches='tight', dpi=600)
pl.savefig('perf.svg', bbox_inches='tight', dpi=600)
pl.savefig('perf.eps', bbox_inches='tight', dpi=600)


#for i in xrange(params['multi_n']):
#    pl.figure(333+i)
#    pl.subplot(211)
#    pl.title('D1 run '+str(i))
#    pl.plot(wd1[i])
#    pl.subplot(212)
#    pl.title('D2 run '+str(i))
#    pl.plot(wd2[i])

#pl.figure(330)
#for i in xrange(params['multi_n']):
#    pl.plot(perf[i], label=i)
pl.show()



