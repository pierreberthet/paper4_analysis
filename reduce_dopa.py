import nest
import nest.raster_plot
import numpy as np
import pylab as pl


nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True})



sim_time = 0.
weight = []



if (not 'bcpnn_dopamine_synapse' in nest.Models()):
    #nest.Install('ml_module')
    nest.Install('/media/backup/temp_milner/save/17.10.14/modules/from.git/bcpnndopa_module/lib/nest/ml_module')


dopa = nest.Create('iaf_neuron', 200)
vt_dopa = nest.Create('volume_transmitter', 1)

nest.ConvergentConnect(dopa, vt_dopa, weight= 5., delay = 1.)

sample_size = 20
pre = nest.Create('iaf_cond_alpha_bias', sample_size)
post = nest.Create('iaf_cond_alpha_bias', sample_size)
poisson_pre = nest.Create('poisson_generator',1)
poisson_post = nest.Create('poisson_generator',1)
poisson_dopa = nest.Create('poisson_generator',1)
poisson_noise = nest.Create('poisson_generator',1)
nest.DivergentConnect(poisson_noise,pre , weight=1., delay=1.)
nest.DivergentConnect(poisson_noise,post , weight=1., delay=1.)
nest.DivergentConnect(poisson_noise,dopa , weight=1., delay=1.)
nest.SetStatus(poisson_noise, {'rate':1800.})

recorder = nest.Create('spike_detector',1)
voltmeter = nest.Create('multimeter', 1, params={'record_from': ['V_m'], 'interval' :0.1} )
nest.SetStatus(voltmeter, [{"to_file": True, "withtime": True, 'label' : 'volt'}])




time = 300.

key = 'C_m'
spread = .2

params = {
        'b': 1.,
        'delay':1.,
        'dopamine_modulated':True,
        'complementary':False,
        'fmax': 20.,
        'gain': 2.,
        'gain_dopa': 1.,
        'n': 0.07,
        'p_i': .01,
        'p_j': .01,
        'p_ij': .00012,
        'tau_i': 5.,
        'tau_j': 6.,
        'tau_e': 40.,
        'tau_p': 200.,
        'tau_n': 100.,
        'value': 1.,
        'k_pow':3.,
        'reverse': 1.
        }



nest.SetDefaults('bcpnn_dopamine_synapse', {'vt':vt_dopa[0]})


default = nest.GetStatus([post[0]], key)[0]
print 'Default value for ', key, 'is ', default
start = (1-spread)*default
end= (1+spread)*default
value = np.arange(start, end, (end-start)/sample_size)

for i in xrange(sample_size):
    nest.SetStatus([post[i]], {key:value[i]})
    
    
nest.DivergentConnect(poisson_pre, pre, weight=4., delay=1.)
nest.DivergentConnect(poisson_post, post, weight=4., delay=1.)
nest.DivergentConnect(poisson_dopa, dopa, weight=4., delay=1.)
nest.ConvergentConnect(post, recorder)
nest.ConvergentConnect(voltmeter, post)
nest.SetStatus(poisson_pre, {'rate': 0.})

nest.CopyModel('bcpnn_dopamine_synapse', 'test', params)
nest.DivergentConnect(pre, post, model='test' )

conn = nest.GetConnections(pre, post)

def simul(pre_rate, post_rate, dopa_rate, duration):

    nest.SetStatus(poisson_pre, {'rate': pre_rate})
    nest.SetStatus(poisson_post, {'rate': post_rate})
    nest.SetStatus(poisson_dopa, {'rate': dopa_rate})
    global sim_time
    global weight
    sim_time+= duration
    nest.Simulate(duration)
    weight.append(np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]))


step=250.
simul(1000.,1000.,1000.,step)
simul(2000.,1000.,1000.,step)
simul(2000.,1000.,1000.,step)
simul(3000.,0.,1500.,step)
simul(3000.,0.,1000.,step)



events = nest.GetStatus(voltmeter)[0]['events']
t = events['times']

pl.subplot(211)
pl.plot(t, events['V_m'])
pl.ylabel('Membrane potential [mV]')

pl.subplot(212)
pl.plot(weight)
pl.show()

nest.raster_plot.from_device(recorder, hist=True)
nest.raster_plot.show()


param = [{'C_m': 250.0,
'E_L': -70.0,
'E_ex': 0.0,
'E_in': -85.0,
'I_e': 0.0,
'V_m': -70.0,
'V_reset': -60.0,
'V_th': -55.0,
'archiver_length': 0,
'bias': 0.0,
'epsilon': 0.001,
'fmax': 20.0,
'frozen': False,
'g_L': 16.6667,
'gain': 1.0,
'global_id': 204,
'kappa': 1.0,
'local': True,
'local_id': 204,
'model': 'iaf_cond_alpha_bias',
'parent': 0,
'recordables': ['V_m',
't_ref_remaining',
'g_ex',
'g_in',
'z_j',
'e_j',
'p_j',
'bias',
'epsilon',
'kappa'],
'state': 0,
't_ref': 2.0,
't_spike': -1.0,
'tau_e': 100.0,
'tau_j': 10.0,
'tau_minus': 20.0,
'tau_minus_triplet': 110.0,
'tau_p': 1000.0,
'tau_syn_ex': 0.2,
'tau_syn_in': 2.0,
'thread': 0,
'type': 'neuron',
'vp': 0}]

