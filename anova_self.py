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
import glob
import scipy.stats as stats

def get_weights(folder):
    fparam = folder+'Test/Parameters/simulation_parameters.json'
    f = open(fparam, 'r')
    params = json.load(f)
    
    params['multi_n']+=1
    
    source_rew= folder+params['rewards_multi_fn']+'_'
    rewards = np.zeros((params['multi_n'], params['n_iterations']))
    
    for i in xrange(params['multi_n']):
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

    return rewards

def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"





######################################
######################################


if len(sys.argv)>3: 
    print "Need 1 folder for auto comparison"
    pass


fname = sys.argv[1]+'/'
pd = sys.argv[2]   # 0 = normal   1= PD only last 5 blocks

params=json.load(open(fname+'Test/Parameters/simulation_parameters.json'))


#print 'Do the simulations match? ',  params[:]['n_recordings']==params[:]['n_recordings']

#diff = difflib.ndiff(open(fparam1,'r').readlines(), open(fparam2,'r').readlines())
#print ''.join(diff)


rew = get_weights(fname)
start = 4
startpd = 11
backs = 40 
tos = 20 
backe = 21 
toe = 1 
#shift = start*params[1]['block_len']*params[1]['t_iteration']/params[1]['resolution']
#shift_rew = start*params[1]['block_len']
#shiftpd = startpd*params[1]['block_len']*params[1]['t_iteration']/params[1]['resolution']
#shiftpd_rew = startpd*params[1]['block_len']
#p = len(fname)-1
p = 3
p2 = 5
perfstart= np.zeros(params['multi_n'], dtype=float)
perfend= np.zeros(params['multi_n'], dtype=float)
j=0
#for i in xrange(shift_rew, params1['n_iterations']):
#    r1[j]=sum(rewa[:,i])
#    r2[j]=sum(rewb[:,i])
#    j+=1

#for i in xrange(start, params1['n_blocks']):
if pd:
    for i in xrange(params['multi_n']):
        for q in xrange(startpd, params['n_blocks']):
            perfstart[i]+=sum(rew[i,q*params['block_len']-backs:q*params['block_len']-tos])
            perfend[i]+=  sum(rew[i,q*params['block_len']-backe:q*params['block_len']-toe])
else:
    for i in xrange(params['multi_n']):
        for q in xrange(start, params['n_blocks']):
            perfstart[i]+=sum(rew[i,q*params['block_len']-backs:q*params['block_len']-tos])
            perfend[i]+=  sum(rew[i,q*params['block_len']-backe:q*params['block_len']-toe])



if pd:
    perfstart = perfstart/((params['n_blocks']-startpd)*(backs-tos-1))
    perfend = perfend/((params['n_blocks']-startpd)*(backe-toe))
else:
    perfstart = perfstart/((params['n_blocks']-start)*(backs-tos-1))
    perfend = perfend/((params['n_blocks']-start)*(backe-toe))


print 'PERF'
print  fname
print 'START mean ',  np.mean(perfstart), 'SD=', np.std(perfstart)
print 'END mean ',  np.mean(perfend), 'SD=', np.std(perfend)
print 'T-TEST: ', stats.ttest_ind(perfstart,perfend)
print 'F-TEST: ', stats.f_oneway(perfstart, perfend)
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print '\n'
norm = np.ones(len(perfstart))*.3333
print 'F-TEST start: ', stats.f_oneway(perfstart, norm)
print 'F-TEST end: ', stats.f_oneway(perfend, norm)
print '+++++++++++++++++++++++++++++++'
print '\n'
print '\n'
