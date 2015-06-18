import numpy as np
import json
import matplotlib
#matplotlib.use('Agg')
import pylab as pl
import sys
import pprint as pp
import difflib
#from difflib_data import *
# Plot the different figures for the merged spikes and voltages recordings.
# This file, as the MergeSpikefiles.py should be one level up than Test/..., the output of a simulation.

import scipy.stats as stats

def get_weights(folder):
    fparam = folder+'Test/Parameters/simulation_parameters.json'
    f = open(fparam, 'r')
    params = json.load(f)
    
    params['multi_n']+=1
    
    source_d1 = folder+params['weights_d1_multi_fn']+'_'
    source_d2 = folder+params['weights_d2_multi_fn']+'_'
    source_rew= folder+params['rewards_multi_fn']+'_'
    source_rp = folder+params['weights_rp_multi_fn']+'_'
    
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
    
    
    #for i in xrange(lend1):
    #    for j in xrange(params['n_actions']):
    #        wd1_m[i,j] = np.mean(wd1[:,i,j])
    #        wd1_std[i,j] = np.std(wd1[:,i,j])
    #        wd2_m[i,j] = np.mean(wd2[:,i,j])
    #        wd2_std[i,j] = np.std(wd2[:,i,j])
    #for i in xrange(lenrp):
    #    for j in xrange(params['n_actions']*params['n_states']):
    #        wrp_m[i,j] = np.mean(wrp[:,i,j])
    #        wrp_std[i,j] = np.std(wrp[:,i,j])

    return wd1, wd2, wrp, rewards
######################################
######################################


if len(sys.argv)<2: 
    print "Need 2 folders for comparison"
    pass


file1 = sys.argv[1]+'/'
file2 = sys.argv[2]+'/'

fparam1 = file1+'Test/Parameters/simulation_parameters.json'
f1 = open(fparam1, 'r')
params1 = json.load(f1)
fparam2 = file2+'Test/Parameters/simulation_parameters.json'
f2 = open(fparam2, 'r')
params2 = json.load(f2)

print 'Do the simulations match? ',  params2['n_recordings']==params1['n_recordings']

#diff = difflib.ndiff(open(fparam1,'r').readlines(), open(fparam2,'r').readlines())
#print ''.join(diff)


wd1a, wd2a, wrpa, rewa = get_weights(file1)
wd1b, wd2b, wrpb, rewb = get_weights(file2)

start = 5
shift = start*params1['block_len']*params1['t_iteration']/params1['resolution']
shift_rew = start*params1['block_len']
a1 = np.zeros(params1['n_recordings']-shift)
b1 = np.zeros(params1['n_recordings']-shift)
c1 = np.zeros(params1['n_recordings']-shift)
a2 = np.zeros(params2['n_recordings']-shift)
b2 = np.zeros(params2['n_recordings']-shift)
c2 = np.zeros(params2['n_recordings']-shift)
#r1 = np.zeros(params1['n_iterations']-shift_rew)
#r2 = np.zeros(params2['n_iterations']-shift_rew)

r1 = np.zeros(params1['multi_n'])
r2 = np.zeros(params2['multi_n'])


j=0
for i in xrange(int(shift),int(params1['n_recordings'])):
    #for multi in xrange(params['multi_n']):
    #a1[j] =sum(sum(abs(wd1a[:, i,:])))
    #b1[j] =sum(sum(abs(wd2a[:, i,:])))
    #c1[j] =sum(sum(abs(wrpa[:, i,:])))
    #a2[j] =sum(sum(abs(wd1b[:, i,:])))
    #b2[j] =sum(sum(abs(wd2b[:, i,:])))
    #c2[j] =sum(sum(abs(wrpb[:, i,:])))
    for q in xrange(params1['multi_n']):
        for k in xrange(params1['n_actions']):
            a1[j] += abs(wd1a[q,i,k] - wd1a[q,i-1,k])
            b1[j] += abs(wd2a[q,i,k] - wd2a[q,i-1,k])
            c1[j] += abs(wrpa[q,i,k] - wrpa[q,i-1,k])
            a2[j] += abs(wd1b[q,i,k] - wd1b[q,i-1,k])
            b2[j] += abs(wd2b[q,i,k] - wd2b[q,i-1,k])
            c2[j] += abs(wrpb[q,i,k] - wrpb[q,i-1,k])
    j+=1

j=0
#for i in xrange(shift_rew, params1['n_iterations']):
#    r1[j]=sum(rewa[:,i])
#    r2[j]=sum(rewb[:,i])
#    j+=1

#for i in xrange(start, params1['n_blocks']):
for i in xrange(params1['multi_n']):
    for q in xrange(start, params1['n_blocks']):
        r1[j]+=sum(rewa[i,q*params1['block_len']-6:q*params1['block_len']-1])
        r2[j]+=sum(rewb[i,q*params1['block_len']-6:q*params2['block_len']-1])
    j+=1

#r1 = r1/params1['multi_n']
#r2 = r2/params2['multi_n']
r1 = r1/((params1['n_blocks']-start)*5.) 
r2 = r2/((params2['n_blocks']-start)*5.)

print 'D1'
print 'mean A', np.mean(a1), 'standard deviation A', np.std(a1)
print 'mean B', np.mean(a2), 'standard deviation B', np.std(a2)
print stats.ttest_ind(a1,a2)

print 'D2'
print 'mean A', np.mean(b1), 'standard deviation A', np.std(b1)
print 'mean B', np.mean(b2), 'standard deviation B', np.std(b2)
print stats.ttest_ind(b1,b2)

print 'RP'
print 'mean A', np.mean(c1), 'standard deviation A', np.std(c1)
print 'mean B', np.mean(c2), 'standard deviation B', np.std(c2)
print stats.ttest_ind(c1,c2)

print 'PERF'
print 'mean A', np.mean(r1), 'standard deviation A', np.std(r1)
print 'mean B', np.mean(r2), 'standard deviation B', np.std(r2)
print stats.ttest_ind(r1,r2)
    
    
    
    
    
