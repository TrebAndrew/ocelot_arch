'''
simple optical system 
source -> aperture -> lens -> image 
'''

import logging
import ocelot
import numpy as np
from ocelot.common.globals import *
from ocelot.optics.wave import *
from ocelot.optics.wave import RadiationField, dfl_waistscan
from ocelot.optics.wave import imitate_sase_dfl, wigner_dfl, dfl_waistscan, generate_gaussian_dfl
from ocelot.gui.dfl_plot import plot_dfl, plot_wigner, plot_dfl_waistscan
from ocelot.rad.propagation import * 
from ocelot.rad.optics_elements import *
from ocelot.rad.transfer_function import * 
from ocelot.rad.optics_line import *

from copy import deepcopy
from ocelot import ocelog
#import array
#from ocelot.common.logging import *
_logger = logging.getLogger(__name__) 
#####################################


dfl = RadiationField()
E_pohoton = 200 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(221,221,1),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
        'dgrid':(400e-5,400e-5,35e-6), #(x,y,z) [m] - size of field matrix
        'power_rms':(25e-5,25e-5,3e-6),#(x,y,z) [m] - rms size of the radiation distribution (gaussian)
        'power_center':(0,0,None),     #(x,y,z) [m] - position of the radiation distribution
        'power_angle':(0,0),           #(x,y) [rad] - angle of further radiation propagation
        'power_waistpos':(0,0),        #(Z_x,Z_y) [m] downstrean location of the waist of the beam
        'wavelength':None,             #central frequency of the radiation, if different from xlamds
        'zsep':None,                   #distance between slices in z as zsep*xlamds
        'freq_chirp':0,                #dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
        'en_pulse':None,                #total energy or max power of the pulse, use only one
        'power':1e6,
        }

dfl = generate_dfl(**kwargs);  #Gaussian beam defenition

#plot_dfl(dfl, fig_name='before', phase=1)

#a = Mask()
#d = FreeSpace(l=200, mx=1.5, my=1.5)
#l = ThinLens(fx=25, fy=25)
#app = ApertureEllips(ax=0.001, ay=0.001, cx=0, cy=0)
#app =  ApertureRect(lx=0.001, ly=0.001, cx=0, cy=0)
#
#line = (app)#, l, app)

#lat = OpticsLine(line)

#dfl = propagate(lat, dfl)

#curve_wave = QuadCurvMask(r = 10, plane='xy')
#dfl = curve_wave.apply(dfl)
#QuadCurvMask(r = 10).apply(dfl)
#dfl = MaxwPropogator_m(z0=15, mx=1, my=1).apply(dfl)

#dfl = MaxwPropogator(z0=100).apply(dfl)

#RectApp = RectMask(dfl.shape())
#RectApp.lx = 0.0003
#RectApp.ly = 0.0003
##RectApp.apply(dfl)
#plot_dfl(dfl, fig_name='after0', phase=1)
#
#EllipsApp = EllipsMask(shape = dfl.shape())
#EllipsApp.ax = 0.0003
#EllipsApp.ay = 0.0003
#dfl=EllipsApp.apply(dfl)
#plot_dfl(dfl, fig_name='after1', phase=1)
#



coeff = [0,0,-250,100]


seed_dfl = generate_dfl(1239.8/500*1e-9, shape=(3,3,2001), dgrid=(1e-3,1e-3,400e-6), power_rms=(0.5e-3,0.5e-3,0.5e-6), 
                        power_center=(0,0,None), power_angle=(0,0), power_waistpos=(0,0), #wavelength=[4.20e-9,4.08e-9], 
                        zsep=None, freq_chirp=0, energy=None, power=10e6, debug=1)

SASE_dfl = imitate_sase_dfl(1239.8/500*1e-9,rho=4e-4, shape=(3,3,2001), dgrid=(1e-3,1e-3,400e-6), power_rms=(0.5e-3,0.5e-3,5e-6), 
                        power_center=(0,0,None), power_angle=(0,0), power_waistpos=(0,0),
                        zsep=None, energy=None, power=10e6, debug=1)

#plot_dfl(seed_dfl, fig_name='before_seed', phase=0)
#plot_dfl(SASE_dfl, fig_name='before_SASE', phase=0)

wig = wigner_dfl(seed_dfl)
SASE_wig = wigner_dfl(SASE_dfl)

plot_wigner(wig, fig_name = 'before', plot_moments=0)
plot_wigner(SASE_wig, fig_name = 'SASE_before', plot_moments=0)

#seed_dfl = 
PhaseChirp = PhaseDelayMask(coeff = coeff).apply(seed_dfl)

#dfl_crip_freq(seed_dfl, coeff, E_ph0=None, return_result = 0)
#SASE_dfl = 
PhaseChirp = PhaseDelayMask(coeff = coeff).apply(SASE_dfl)

#dfl_crip_freq(SASE_dfl, coeff, E_ph0=None, return_result = 0)

wig = wigner_dfl(seed_dfl)
SASE_wig = wigner_dfl(SASE_dfl)

#a = calc_dfl_chirp(SASE_dfl)
#b = calc_dfl_chirp(seed_dfl)
#print(a, b)

plot_wigner(wig, fig_name = 'after', plot_moments=0)
plot_wigner(SASE_wig, fig_name = 'SASE_after', plot_moments=0)

#plot_dfl(seed_dfl, fig_name='after_seed', phase=0)
#plot_dfl(SASE_dfl, fig_name='after_SASE', phase=0)

###phase delay check 








