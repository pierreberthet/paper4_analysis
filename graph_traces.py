import numpy as np
import matplotlib
import pylab as pl



sim = 1000

spiketrain_pre = np.zeros(sim)
spiketrain_post = np.zeros(sim)
z_i = np.zeros(sim) 
z_j = np.zeros(sim) 
e_i = np.zeros(sim) 
p_i = np.zeros(sim) 
e_j = np.zeros(sim) 
p_j = np.zeros(sim) 
e_ij = np.zeros(sim) 
p_ij = np.zeros(sim) 

spiketrain_pre[100:120] = 1.
spiketrain_post[103:110] = 1.

spiketrain_pre[200:250] = 1.
spiketrain_post[203:280] = 1.
spiketrain_post[403:480] = 1.

z_i[0] = .1
z_j[0] = .1
e_i[0] = .1
p_i[0] = .1
e_j[0] = .1
p_j[0] = .1
e_ij[0] = .1
p_ij[0] = .1

log = np.zeros(sim) 
w = np.zeros(sim) 
b = np.zeros(sim) 

tauz = 5.
taue = 50.
taup = 1000.
K = -0.1

for i in xrange(1,sim):
    z_i[i] = z_i[i-1] + (spiketrain_pre[i] - z_i[i-1])/tauz
    e_i[i] = e_i[i-1] + (z_i[i] - e_i[i-1])/taue
    p_i[i] = p_i[i-1] + K*(e_i[i] - p_i[i-1])/taup
    z_j[i] = z_j[i-1] + (spiketrain_post[i] - z_j[i-1])/tauz
    e_j[i] = e_j[i-1] + (z_j[i] - e_j[i-1])/taue
    p_j[i] = p_j[i-1] + K*(e_j[i] - p_j[i-1])/taup
    z_i[i] = z_i[i-1] + (spiketrain_pre[i] - z_i[i-1])/tauz
    e_ij[i] = e_ij[i-1] + (z_i[i]*z_j[i] - e_ij[i-1])/taue
    p_ij[i] = p_ij[i-1] + K*(e_ij[i] - p_ij[i-1])/taup
    w[i] = p_ij[i]/(p_i[i]*p_j[i])
    log[i] = np.log(w[i])
    b[i] = np.log(p_j[i])

pl.figure(5)
zi = pl.plot(np.arange(sim), z_i, label="z_i")
zj = pl.plot(np.arange(sim), z_j, label="z_j")
ei = pl.plot(np.arange(sim), e_i, label="e_i")
pi = pl.plot(np.arange(sim), p_i, label="p_i")
ej = pl.plot(np.arange(sim), e_j, label="e_j")
pj = pl.plot(np.arange(sim), p_j, label="p_j")
eij = pl.plot(np.arange(sim), e_ij, label="e_ij")
pij = pl.plot(np.arange(sim), p_ij, label="p_ij")
lopolt = pl.plot(np.arange(sim), log, label="log")
wplot = pl.plot(np.arange(sim), w, label="w")
bplot = pl.plot(np.arange(sim), b, label="bias")
pl.legend()
pl.show()
