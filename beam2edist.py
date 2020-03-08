import sys, os
import matplotlib.pyplot as plt
from ocelot.utils.xfel_utils import *
from ocelot.gui.genesis_plot import *
import time
from ocelot.common.globals import *  #import of constants like "h_eV_s" and "speed_of_light"
from ocelot.common.py_func import *

import numpy as np
import random
"""
Created on Mon Jan  6 20:29:09 2020

@author: andrei
"""
'''
reads BeamArray()
returns GenesisElectronDist()

'''

E_beam = 8.5   # Electon beam energy (e.g.14.4) [GeV]

#parray = generate_parray(nparticles=20000)
#edist = GenesisElectronDist()
#edist = parray2edist(parray)
#plot_edist(edist)
#plt.plot(edist.s)
#print(edist.s)#, edist.s)
#plt.show()
beam = BeamArray()
beam = generate_beam(E=20.0, dE=2.5e-3, I=5000, l_beam=1e-6, emit_n=0.5e-6, beta=20, l_window=6e-6, shape='gaussian', chirp=0.0009)

#beam=edist2beam(edist, step=5000e-7)

coeff = [0.0, +1.0, -0, -0, 2.0]
beam.add_chirp_poly(coeff, s0=None)

plot_beam(beam, showfig=1, savefig=0)
#%%
dist='gaussian' #flattop_quad, flattop_el
rparticles = 10000

P = beam.I/np.sum(beam.I)
pack = np.around(P*rparticles)

edist = GenesisElectronDist()
edist.t = []
edist.g = []

for i in range(beam.len()-1): 
     
    t1 = beam.s[i]/speed_of_light 
    t2 = beam.s[i + 1]/speed_of_light 
    t = np.random.uniform(t1, t2, size=int(pack[i])) #okeish, better to replace by linear splines

    edist.t = np.append(edist.t, t)

    g1 = beam.E[i]/m_e_GeV
    g2 = beam.E[i+1]/m_e_GeV
    
#    g = np.random.uniform(g1, g2, size=int(pack[i])) #uniform distribution 
    g = (t-t1)*(g2 - g1)/(t2- t1) + g1 #linear spline 
    edist.g = np.append(edist.g, g)
    
    x = beam.x[i]
    xp = beam.xp[i]
    
    y = beam.y[i]
    yp = beam.yp[i]
    
    emit_x = beam.emit_x[i]
    emit_y = beam.emit_y[i]
    
    beta_x = beam.beta_x[i]
    beta_y = beam.beta_y[i]
    
    alpha_x = beam.alpha_x[i]
    alpha_y = beam.alpha_y[i]
    
    gamma_x = (1 + alpha_x**2)/beta_x 
    gamma_y = (1 + alpha_y**2)/beta_y
    
    if dist in ['gaussian', 'gaus', 'g']:
        mean_x_xp = [x, xp]
        cov_x_xp = [[emit_x*beta_x, -alpha_x*emit_x],[-alpha_x*emit_x, emit_x*gamma_x]]#not sure about cov matrix
        dist_x, dist_xp = np.random.multivariate_normal(mean_x_xp, cov_x_xp, int(pack[i])).T
        
        edist.x = np.append(edist.x, dist_x)
        edist.xp = np.append(edist.xp, dist_xp)
        
        mean_y_yp = [y, yp]
        cov_y_yp = [[emit_y*beta_y, -alpha_y*emit_y],[-alpha_y*emit_y, emit_y*gamma_y]] #not sure about cov matrix
        dist_y, dist_yp = np.random.multivariate_normal(mean_y_yp, cov_y_yp, int(pack[i])).T
        
        edist.y = np.append(edist.y, dist_y)
        edist.yp = np.append(edist.yp, dist_yp)

    elif dist in ['flattop', 'ft']:
        pass
    elif dist in ['Kapchinsky-Vladimirsky', 'KV']:
        pass
    else: 
        raise ValueError('we did not implement your distribution yet, try "gaussian", "lattop" or "Kapchinsky-Vladimirsky" ("KV") distributions')
        
edist.part_charge = beam.charge()/edist.len()
#
n = np.linspace(1, edist.len(), edist.len())
plt.figure('1')
plt.plot(n, edist.g)
plt.figure('2')
plt.plot(edist.t, edist.g)

plt.show()
#plot_edist(edist)

#%%
#n = 100000
##
#r = np.random.uniform(-1, 1, n)
#y = np.random.uniform(-1, 1, n)
#x = np.random.uniform(-1, 1, n)
#x1 = np.random.uniform(-1, 1, n)
#y1 = np.zeros(n)
#for i in range(len(r)):
#    if r[i] >= 0:  
#        x[i] = np.sqrt(r[i]**2 - y[i]**2)       
#    elif r[i] < 0:  
#        x[i] = -np.sqrt(r[i]**2 - y[i]**2)
##  
#
#        
#    
##
#plt.plot(x, y, 'ro', ms=0.3)
#plt.show()
#
#
#










