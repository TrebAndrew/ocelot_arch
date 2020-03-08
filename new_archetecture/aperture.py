'''
aperture.py for pre pull request check v0.1
'''
import logging
import time

import numpy as np
from copy import deepcopy
from math import factorial

import ocelot
from ocelot.common.globals import *
from ocelot import ocelog
from ocelot.common.ocelog import *
_logger = logging.getLogger(__name__) 

from ocelot.common.globals import *
'''
file for appertures
'''

from ocelot.optics.wave import *
from ocelot.optics.wave import RadiationField, dfl_waistscan
from ocelot.optics.wave import imitate_sase_dfl, wigner_dfl, dfl_waistscan, generate_gaussian_dfl
from ocelot.gui.dfl_plot import plot_dfl, plot_wigner, plot_dfl_waistscan
from ocelot.rad.propagation import * 
from ocelot.rad.optics_elements import *
from ocelot.rad.transfer_function import * 
from ocelot.rad.optics_line import *


### goes to optics_line.py ### 
flatten = lambda *n: (e for a in n
                      for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))

class OpticsLine():
    def __init__(self, sequence, start=None, stop=None):
        self.sequence = list(flatten(sequence))
        self.stop = stop
        self.start = start
        self.update_optics_masks()

    def update_optics_masks(self):
        print("update mask")

        for element in self.sequence:
            print(element)

            get_transfer_function(element)

    def estimate_mesh(self):
        for element in self.sequence:
            element.mesh = 0
            
def get_transfer_function(element):
    element.mask = Mask()
    #better to implement via 'try'?
    if element.__class__ is None:
        raise ValueError('Optics element must belong to the OpticsElement class')
    
    elif element.__class__ is ApertureRect:
        mask = RectMask()
        mask.lx = element.lx
        mask.ly = element.ly
        element.mask = mask
        
    elif element.__class__ is ApertureEllips:
        mask = EllipsMask()
        mask.ax = element.ax
        mask.ay = element.ay
        element.mask = mask
  
### goes to optics_element.py
class Aperture(OpticsElement):
    """
    Aperture
    """

    def __init__(self, eid=None):
        OpticsElement.__init__(self, eid=eid)
        
class ApertureRect(Aperture):
    """
    Rectangular aperture
    """

    def __init__(self, lx=np.inf, ly=np.inf, cx=0., cy=0., eid=None):
        Aperture.__init__(self, eid=eid)
        self.lx = lx
        self.ly = ly
        self.cx = cx
        self.cy = cy


class ApertureEllips(Aperture):
    """
    Elliptical Aperture
    """

    def __init__(self, ax=np.inf, ay=np.inf, cx=0., cy=0., eid=None):
        Aperture.__init__(self, eid=eid)
        self.ax = ax
        self.ay = ay
        self.cx = cx
        self.cy = cy

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "

class RectMask(Mask):
    def __init__(self, shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.lx = np.inf
        self.ly = np.inf
        self.cx = 0
        self.cy = 0
        self.mask = None

    def apply(self, dfl):
        
        if self.mask is None:
            self.get_mask(dfl)
        mask_idx = np.where(self.mask == 0)

        dfl_energy_orig = dfl.E()
        dfl.fld[:, mask_idx[0], mask_idx[1]] = 0

        if dfl_energy_orig == 0:
            _logger.warn(ind_str + 'dfl_energy_orig = 0')
        elif dfl.E() == 0:
            _logger.warn(ind_str + 'done, %.2f%% energy lost' % (100))
        else:
            _logger.info(ind_str + 'done, %.2f%% energy lost' % ((dfl_energy_orig - dfl.E()) / dfl_energy_orig * 100))
        return dfl

    def get_mask(self, dfl):
        """
        model rectangular aperture to the radaition in either domain
        """
        _logger.info('applying square aperture to dfl')
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.shape = dfl.shape()
            
        if np.size(self.lx) == 1:
            self.lx = [-self.lx / 2, self.lx / 2]
        if np.size(self.ly) == 1:
            self.ly = [-self.ly / 2, self.ly / 2]
        _logger.debug(ind_str + 'ap_x = {}'.format(self.lx))
        _logger.debug(ind_str + 'ap_y = {}'.format(self.ly ))

        idx_x = np.where((self.x() >= self.lx[0]) & (self.x() <= self.lx[1]))[0]
        idx_x1 = idx_x[0]
        idx_x2 = idx_x[-1]

        idx_y = np.where((self.y() >= self.ly [0]) & (self.y() <= self.ly [1]))[0]
        idx_y1 = idx_y[0]
        idx_y2 = idx_y[-1]

        _logger.debug(ind_str + 'idx_x = {}-{}'.format(idx_x1, idx_x2))
        _logger.debug(ind_str + 'idx_y = {}-{}'.format(idx_y1, idx_y2))

        self.mask = np.zeros_like(dfl.fld[0, :, :])
        self.mask[idx_y1:idx_y2, idx_x1:idx_x2] = 1
        return self.mask

    def __mul__(self, other):
        m = RectMask()
        if other.__class__ in [RectMask, ] and self.mask is not None and other.mask is not None:
            m.mask = self.mask * other.mask
            return m

class EllipsMask(Mask):   
    def __init__(self, shape=(0,0,0)):
        Mask.__init__(self, shape=shape)
        self.ax = np.inf
        self.ay = np.inf
        self.cx = 0
        self.cy = 0
        self.mask = None 
    
    def ellipse(self, dfl):    
        x, y = np.meshgrid(self.x(), self.y())
        xp =  (x - self.cx)*np.cos(pi) + (y - self.cy)*np.sin(pi)
        yp = -(x - self.cx)*np.sin(pi) + (y - self.cy)*np.cos(pi)
        return (2*xp/self.ax)**2 + (2*yp/self.ay)**2
    
    def apply(self, dfl):
        """
        apply elliptical aperture to the radaition in either domain
        """
        
        _logger.info('applying elliptical aperture to dfl')
        _logger.debug(ind_str + 'ap_x = {}'.format(self.ax) + 'cx = {}'.format(self.cx))
        _logger.debug(ind_str + 'ap_y = {}'.format(self.ay) + 'cy = {}'.format(self.cy))

        
        if self.mask is None:
            self.get_mask(dfl)
        
        mask_idx = np.where(self.mask == 0)

        dfl_energy_orig = dfl.E()

        dfl.fld[:, mask_idx[0], mask_idx[1]] = 0
        
        if dfl_energy_orig == 0:
            _logger.warn(ind_str + 'dfl_energy_orig = 0')
        elif dfl.E() == 0:
            _logger.warn(ind_str + 'done, %.2f%% energy lost' % (100))
        else:
            _logger.info(ind_str + 'done, %.2f%% energy lost' % ((dfl_energy_orig - dfl.E()) / dfl_energy_orig * 100))
            
        return dfl
    
    def get_mask(self, dfl):
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.shape = dfl.shape()
        
        self.mask = np.zeros_like(dfl.fld[0, :, :])
        self.mask[self.ellipse(dfl) <= 1] = 1
        return self.mask
        
        


### main body

E_pohoton = 200 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(421,421,1),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
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
#plot_dfl(dfl, fig_name='before', phase=0)

###
#RectApp = RectMask(dfl.shape())
#RectApp.lx = 0.0005
#RectApp.ly = 0.0005
#RectApp.apply(dfl)
###
#EllipsApp = EllipsMask(shape = dfl.shape())
#EllipsApp.ax = 0.0005
#EllipsApp.ay = 0.0005
#dfl=EllipsApp.apply(dfl)
###

RectApp = ApertureRect(lx=0.0005, ly=0.0005, cx=0, cy=0)
EllipsApp = ApertureEllips(ax=0.0006, ay=0.0006, cx=0, cy=0)

line = (RectApp, EllipsApp)
lat = OpticsLine(line)
#
dfl = propagate(lat, dfl)                
plot_dfl(dfl, fig_name='after', phase=0)



























