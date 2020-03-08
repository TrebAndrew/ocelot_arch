#!/usr/bin/env python3
import logging
_logger = logging.getLogger(__name__) 

from ocelot.gui.genesis_plot import *
from ocelot.optics.new_wave import RadiationField, generate_gaussian_dfl


import numpy as np
from copy import deepcopy
from numpy import inf, complex128, complex64
from math import factorial

try:
    import pyfftw
    fftw_avail = True
except ImportError:
    print("wave.py: module PYFFTW is not installed. Install it if you want speed up dfl wavefront calculations")
    fftw_avail = False

#%%
#optics elements check
dfl = RadiationField()
E_pohoton = 1239.8#200 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(101,101,51),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
        'dgrid':(400e-5,400e-5,35e-6), #(x,y,z) [m] - size of field matrix
        'power_rms':(25e-5,25e-5,4e-6),#(x,y,z) [m] - rms size of the radiation distribution (gaussian)
        'power_center':(0,0,None),     #(x,y,z) [m] - position of the radiation distribution
        'power_angle':(0,0),           #(x,y) [rad] - angle of further radiation propagation
        'power_waistpos':(0,0),        #(Z_x,Z_y) [m] downstrean location of the waist of the beam
        'wavelength':None,             #central frequency of the radiation, if different from xlamds
        'zsep':None,                   #distance between slices in z as zsep*xlamds
        'freq_chirp':0,                #dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
        'en_pulse':None,                #total energy or max power of the pulse, use only one
        'power':1e6,
        }

dfl = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition
#%%
fig_text = 'plot_xz'
figsize = 3
plot_proj = True
plot_slice = True
log_scale = False
cmap='Greys'
x_units = 'mm'
y_units = 'mm'
E_ph_units = 'eV'
_logger.info('plotting x vs. energy distribution')

I = dfl.intensity()
I_units_phsmmbw='$\gamma/s/mm^2/0.1\%bw$'
I_units_phsbw='$\gamma/s/0.1\%bw$'
x = dfl.scale_x()
if x_units == 'mm':
    x = x * 1e3
    x_label_txt = 'x, [mm]' 
elif x_units == 'um':
    x = x * 1e6
    x_label_txt = '$x, [\mu m]$'
else:
    raise ValueError('Incorrect units error')

y = dfl.scale_y()
if y_units == 'mm':
    y = y * 1e3
elif y_units == 'um':
    y = y * 1e6
else:
    raise ValueError('Incorrect units')
    
if E_ph_units in ['eV', 'ev']:
    E_ph = h_eV_s * dfl.grid_w()
    E_label_txt = '$E_{photon}$ [eV]'    
elif y_units in ['keV', 'kev']:
    E_ph = E_ph = h_eV_s * dfl.grid_w() * 1e-3
    E_label_txt = '$E_{photon}$ [keV]'    
else:
    raise ValueError('Incorrect units')


fig = plt.figure(fig_text)
plt.clf()
fig.set_size_inches((4.5 * figsize, 3.25 * figsize), forward=True)

if plot_proj:
    #definitions for the axes
    left, width = 0.18, 0.57
    bottom, height = 0.14, 0.55
    left_h = left + width + 0.02 - 0.02
    bottom_h = bottom + height + 0.02 - 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.15, height]

    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    axScatter = plt.axes(rect_scatter, sharex=axHistx, sharey=axHisty)
else:
    axScatter = plt.axes()

if log_scale != 0:
    if log_scale==1: 
        log_scale=0.01
    I_lim = np.amax(I)
    axScatter.pcolormesh(E_ph, x, I[:,dfl.Ny() // 2,:].T, cmap=cmap, 
                                         norm=colors.SymLogNorm(linthresh=I_lim * log_scale, linscale=2,
                                              vmin=-I_lim, vmax=I_lim),
                                                                vmax=I_lim, vmin=-I_lim)
else:
    axScatter.pcolormesh(E_ph, x, I[:,dfl.Ny() // 2,:].T, cmap=cmap)

axScatter.set_ylabel(x_label_txt, fontsize=18)
axScatter.set_xlabel(E_label_txt, fontsize=18)
if plot_slice:
    
    I_xy = I[:, dfl.Nx() // 2, dfl.Ny() // 2]
    I_x  = I[dfl.Nz() // 2, :, dfl.Ny() // 2]
    
    axHistx.plot(E_ph, I_xy, color='blue')
    axHisty.plot(I_x, x, color='blue')
    
    axHisty.set_xlabel(I_units_phsmmbw, fontsize=18, color='blue')
    axHistx.set_ylabel(I_units_phsmmbw, fontsize=18, color='blue')
    
#    ax2.set_ylabel(I_units_phsbw, fontsize=18, color='black')
    
    axScatter.axis('tight')
      
    for tl in ax1.get_xticklabels():
        tl.set_visible(False)
    for tl in ax1.get_yticklabels():
        tl.set_visible(False)
    ax2.tick_params(labelright='off')
    
    xticks = axHistx.yaxis.get_major_ticks()
    xticks[1].set_visible(False)
    
if plot_proj:
    I_xy = np.sum(dfl.intensity(), axis=(1, 2))
    
    ax1 = axHistx.twinx()
    ax1.plot(E_ph, I_xy, '--',color='black')
    ax1.set_ylabel(I_units_phsbw, fontsize=18, color='black')

    axScatter.axis('tight')

  
    for tl in axHistx.get_xticklabels():
        tl.set_visible(False)
    for tl in axHisty.get_yticklabels():
        tl.set_visible(False)
        
plt.show()
#%%

















