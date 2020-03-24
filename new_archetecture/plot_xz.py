#!/usr/bin/env python3
import logging
_logger = logging.getLogger(__name__) 

from ocelot.gui.genesis_plot import *
from ocelot.optics.wave import RadiationField, generate_gaussian_dfl

import pickle

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
def plot_dfl_xz(dfl, whichis_derectrion='x', plot_proj=True, plot_slice=True, E_slice=None, log_scale=False,
                x_units = 'mm', E_ph_units = 'eV', figsize = 3,
                cmap='Greys', fig_name='dfl_plot_xz', showfig=True, savefig=False):

    start_time = time.time()

    _logger.info('plotting {} vs. energy distribution'.format(whichis_derectrion))

    filePath = dfl.filePath
    
    if dfl.domain_z == 't':#make it universal for both domains
        dfl.to_domain('f')
    if dfl.domain_xy == 'k': #make it universal for both domains
        dfl.to_domain('s')
        
    I = dfl.intensity()
    
    I_units_phsmmbw='$\gamma/s/mm^2/0.1\%bw$'
    I_units_phsbw='$\gamma/s/0.1\%bw$'
    
    if whichis_derectrion == 'x':
        x = dfl.scale_x()
        I_xz = I[:,dfl.Ny() // 2,:].T
        if x_units == 'mm':
            x = x * 1e3
            x_label_txt = 'x, [mm]' 
        elif x_units == 'um':
            x = x * 1e6
            x_label_txt = '$x, [\mu m]$'
        else:
            raise ValueError('Incorrect units error')
    elif whichis_derectrion == 'y':
        x = dfl.scale_y()
        I_xz = I[:,:,dfl.Ny() // 2].T
        if x_units == 'mm':
            x = x * 1e3
            x_label_txt = 'y, [mm]' 
        elif x_units == 'um':
            x = x * 1e6
            x_label_txt = '$y, [\mu m]$'
        else:
            raise ValueError('Incorrect units error')
    else:
        raise ValueError('"whichis_derectrion" must be "x" or "y"')

    E_ph = h_eV_s * dfl.scale_kz() * speed_of_light/2/np.pi                 
    if E_ph_units in ['eV', 'ev']:
        E_label_txt = '$E_{\gamma}$ [eV]'    
    elif y_units in ['keV', 'kev']:
        E_ph = E_ph * 1e-3
        E_label_txt = '$E_{photon}$ [keV]'    
    else:
        raise ValueError('Incorrect units')
    
    if E_slice is None:
        E_slice_idx = dfl.Nz()//2
    elif E_slice == 'max':
        E_slice_idx = np.argmax(I[:,dfl.Ny()//2,dfl.Nx()//2], axis=0)
    elif E_slice >= np.min(E_ph) and E_slice <= np.max(E_ph):
        E_slice_idx = min(range(len(E_ph)), key=lambda i: abs(E_ph[i]-E_slice))
    else:
        raise ValueError('Incorrect "E_slice" value, it can be None or some photon energy in the [{},{}] range or "max"'.format(round(np.min(E_ph)),round(np.max(E_ph))))
    if E_slice_idx is not None:
        E_slice = E_ph[E_slice_idx]
        print(E_slice)
    
    fig = plt.figure(fig_name)
    plt.clf()
    fig.set_size_inches((4.5 * figsize, 3.25 * figsize))#, forward=True)
    
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
    
    
    if log_scale is True: 
        log=0.01
        I_lim = np.amax(I)
        axScatter.pcolormesh(E_ph, x, I_xz, cmap=cmap, 
                                             norm=colors.SymLogNorm(linthresh=I_lim * log, linscale=2,
                                                  vmin=-I_lim, vmax=I_lim),
                                                                    vmax=I_lim, vmin=-I_lim)
    else:
        axScatter.pcolormesh(E_ph, x, I_xz, cmap=cmap)
    
    axScatter.set_ylabel(x_label_txt, fontsize=18)
    axScatter.set_xlabel(E_label_txt, fontsize=18)

    if plot_slice:
        
        I_xy = I[:, dfl.Nx() // 2, dfl.Ny() // 2]
        if whichis_derectrion == 'x':
            I_x  = I[E_slice_idx, :, dfl.Ny() // 2]
        elif whichis_derectrion == 'y':
            I_x  = I[E_slice_idx, dfl.Ny() // 2, :]
        else:
            raise ValueError('"whichis_derectrion" must be "x" or "y"')

        axHistx.plot(E_ph, I_xy, color='blue', label="on-axis, \nxy slise at\n{} eV".format(round(E_slice)))
        axHisty.plot(I_x, x, color='blue')
        axHistx.legend(bbox_to_anchor=(1.08, 1), loc='upper left', borderaxespad=0.)
        
        
        if log_scale is True:
            axHistx.set_yscale('log')
            axHisty.set_xscale('log')
        else:
            axHistx.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0,0))
            axHisty.ticklabel_format(style='sci', axis='x', useOffset=True, scilimits=(0,0))
      
        axHisty.set_xlabel(I_units_phsmmbw, fontsize=18, color='blue')
        axHistx.set_ylabel(I_units_phsmmbw, fontsize=18, color='blue')
        axHisty.xaxis.labelpad = 20
        axHistx.yaxis.labelpad = 20
       
        axScatter.axis('tight')
        
        xticks = axHistx.yaxis.get_major_ticks()
        xticks[1].set_visible(False)
        
    if plot_proj:
        I_xy = np.sum(dfl.intensity(), axis=(1, 2))*dfl.dx*dfl.dy * (1e3)**2
        
        ax1 = axHistx.twinx()
        ax1.plot(E_ph, I_xy, '--',color='black', label="integrated")
        ax1.legend(bbox_to_anchor=(1.08, 0.5), loc='upper left', borderaxespad=0.)
#                     horizontalalignment='left', verticalalignment='top', fontsize=14, rotation=90)
        ax1.set_ylabel(I_units_phsbw, fontsize=18, color='black')
 
        if log_scale is True:
            ax1.set_yscale('log')
        else:
            ax1.ticklabel_format(style='sci', axis='y', useOffset=True, scilimits=(0,0))
    
        axScatter.axis('tight')
        
        for tl in axHistx.get_xticklabels():
            tl.set_visible(False)
        for tl in axHisty.get_yticklabels():
            tl.set_visible(False)
         
    plt.draw()
    if savefig != False:
        if savefig == True:
            savefig = 'png'
            print('here')
        _logger.debug(ind_str + 'saving *.{:}'.format(savefig))
        fig.savefig(filePath + fig_name + '.' + str(savefig), format=savefig)
    _logger.debug(ind_str + 'done in {:.2f} seconds'.format(time.time() - start_time))
    
    plt.draw()
    if showfig == True:
        _logger.debug(ind_str + 'showing dfl')
        rcParams["savefig.directory"] = os.path.dirname(filePath)
        plt.show()
    else:        
        plt.close(fig)

    plt.show()

#%%
#optics elements check
dfl = RadiationField()
E_pohoton = 1239.8#200 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(101,101,51),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
        'dgrid':(400e-5,400e-5,35e-6), #(x,y,z) [m] - size of field matrix
        'power_rms':(25e-5,35e-5,1e-6),#(x,y,z) [m] - rms size of the radiation distribution (gaussian)
        'power_center':(0,0,None),     #(x,y,z) [m] - position of the radiation distribution
        'power_angle':(0,0),           #(x,y) [rad] - angle of further radiation propagation
        'power_waistpos':(0,0),        #(Z_x,Z_y) [m] downstrean location of the waist of the beam
        'wavelength':None,             #central frequency of the radiation, if different from xlamds
        'zsep':None,                   #distance between slices in z as zsep*xlamds
        'freq_chirp':0,                #dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
        'en_pulse':None,                #total energy or max power of the pulse, use only one
        'power':1e6,
        }

print('extracting wfr from files')
#%%
wfrPathName='/home/andre/Documents/1_term_master_s/Sec_und_paper/code/fields/'
wfr1FileName = 'segmented_undulator.scr'

afile = open(wfrPathName + wfr1FileName, 'rb')
screen   =  pickle.load(afile)
afile.close()
dfl = screen2dfl(screen, polarization='x', current=0.4, gamma=3.0/m_e_GeV)

dfl.filePath = '/home/andre/Documents/1_term_master_s/Sec_und_paper/tex/v0.8'
#dfl = generate_gaussian_dfl(**kwargs)  #Gaussian beam defenitio
#%%
plot_dfl_xz(dfl, E_slice=5000, savefig=True, fig_name='spec_sec_far_no_ap.pdf')
plt.savefig('/home/andre/Documents/1_term_master_s/Sec_und_paper/tex/v0.8/spec_sec_far_no_ap.pdf')














