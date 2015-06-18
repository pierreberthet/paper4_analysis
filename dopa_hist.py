import numpy as np
import json
import pylab as pl
import collections
import scipy.interpolate as sp

fparam = 'Test/Parameters/simulation_parameters.json'
f = open(fparam, 'r')
params = json.load(f)

recorder_type = 'spikes'
cell = 'rew'
data = np.loadtxt(params['spiketimes_folder']+cell+'_merged_'+recorder_type+'.dat' )

outcome = np.loadtxt(params['rewards_multi_fn']+'_0')


spread = 2
x = len(outcome) * spread 
tempd =np.sort(data[:,1])[params['t_init']:] 

distrib = np.histogram(tempd, x)[0]
#distrib = np.histogram(data[:,1], x)[0]

#distrib = distrib[0] / float(np.sum(distrib[0]))
#distrib = distrib - np.mean(distrib)


adapt = np.max(distrib)

pl.figure(321)
routcome = np.zeros(x)
for i in xrange(len(outcome)):
    routcome[spread*i:spread*i+spread] = outcome[i] * adapt


y=np.mean(distrib)
#pl.title('dopamine dynamic')
pl.plot(routcome)
pl.plot(distrib, c='k')
pl.fill_between(np.arange(x), distrib, y, where=distrib>=y, facecolor='green', interpolate=True)
pl.fill_between(np.arange(x), distrib, y, where=distrib<=y, facecolor='red', interpolate=True)



#pl.figure(322)
#rpe_smooth = sp.interp1d(np.arange(0,x),distrib, kind='cubic' )
#points = np.linspace(0,x-1,5*x)
#smoothed_distrib = rpe_smooth(points)
#pl.plot(points, smoothed_distrib)
#y=np.mean(smoothed_distrib)
#pl.fill_between(points, smoothed_distrib, y, where=smoothed_distrib>=y, facecolor='green', interpolate=True)
#pl.fill_between(points, smoothed_distrib, y, where=smoothed_distrib<=y, facecolor='red', interpolate=True)


pl.figure(654)
vt = np.zeros(params['t_sim']/params['dt'])
tracevt= np.zeros(params['t_sim']/params['dt'])
for x,y in collections.Counter(tempd).items():
    vt[x*10]=y
for i in xrange(int(params['params_dopa_bcpnn']['tau_n']), int( params['t_sim']/params['dt'] ) ):
    tracevt[i]=tracevt[i-1] + (vt[i]- tracevt[i-1])/params['params_dopa_bcpnn']['tau_n']

pl.plot(tracevt)

#pl.figure(654)
#tracevt = tracevt / params['num_rew_neurons']
#pl.plot(tracevt)

pl.show()
#print outcome



