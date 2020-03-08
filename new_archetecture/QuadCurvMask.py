'''
propagation_mask.py for pre pull request check v0.1
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
    
    elif element.__class__ is ThinLens:
        element.mask = LensMask(element.fx, element.fy)
  
### goes to optics_element.py
class ThinLens(OpticsElement):
    """
    Lens element
    """
    def __init__(self, fx=np.inf, fy=np.inf, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.fx = fx
        self.fy = fy

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "
class QuadCurvMask(Mask):
    def __init__(self, r=np.inf, plane='x', shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.r = r # the radius of curvature
        self.plane = plane # the plane in which the curvature is applied
        self.mask = None # the transfer function itself
        self.domains = 's' # the domain in which the transfer function is calculated
        self.domain_z = None # the domain in which wavefront curvature is introduced 

    def apply(self, dfl):
        
        _logger.debug('curving radiation wavefront by {}m in {} domain'.format(self.r, self.domain_z))
            
        if self.mask is None:
            self.get_mask(dfl) 
    
        dfl.to_domain(self.domain_z + 's') #the field must be in 's' domain
        dfl.fld *= self.mask 
        dfl.to_domain(domains) #return to the original domain

        return dfl
    
    def get_mask(self, dfl):
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.xlamds = dfl.xlamds
        self.shape = dfl.shape()
        
#        domains = dfl.domain_z, dfl.domain_xy

        if self.domain_z is None:
            self.domain_z = dfl.domain_z
        
        if np.size(self.r) == 1:
            if self.r == 0:
                raise ValueError('radius of curvature should not be zero')
            elif self.r == np.inf:
                _logger.debug(ind_str + 'radius of curvature is infinite, skipping')
                return
            else:
                pass
            
        x, y = np.meshgrid(self.x(), self.y())
        if self.plane == 'xy' or self.plane == 'yx':
            arg2 = x ** 2 + y ** 2
        elif self.plane == 'x':
            arg2 = x ** 2
        elif self.plane == 'y':
            arg2 = y ** 2
        else:
            raise ValueError('"plane" should be in ["x", "y"]')

        if self.domain_z == 'f':
            k = 2 * np.pi /self.kz()

            if np.size(self.r) == 1:
                self.mask = np.exp(-1j * k[:, np.newaxis, np.newaxis] * arg2[np.newaxis, :, :] / self.r / 2) #H = exp(-i * k / 2 * (x^2 + y^2))
            elif np.size(self.r) == dfl.Nz():
                self.mask = np.exp(-1j * k[:, np.newaxis, np.newaxis] * arg2[np.newaxis, :, :] / self.r[:, np.newaxis, np.newaxis] / 2)
            else:
                raise ValueError('wrong dimensions of radius of curvature')    

        elif self.domain_z == 't':
            k = 2 * np.pi / self.xlamds

            if np.size(self.r) == 1:
                self.mask = np.exp(-1j * k * arg2 / self.r / 2)
            elif np.size(self.r) == dfl.Nz():
                self.mask = np.exp(-1j * k * arg2[np.newaxis, :, :] / self.r[:, np.newaxis, np.newaxis] / 2)
            else:
                raise ValueError('wrong dimensions of radius of curvature')
        else:
            raise ValueError('domain_z should be in ["f", "t", None]')

        return self.mask
    
class LensMask(Mask):
    def __init__(self, fx=np.inf, fy=np.inf, shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.fx = fx
        self.fy = fy
        self.mask = None
              
    def apply(self, dfl):
        _logger.info('apply the lens mask')
        domains = dfl.domains()

        if self.mask is None:
            self.get_mask(dfl)
                
        dfl.to_domain('fs')
        dfl.fld *= self.mask
        dfl.to_domain(domains)
        
        return dfl
    
    def get_mask(self, dfl):

        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.xlamds = dfl.xlamds
        self.shape = dfl.shape()

        _logger.info('get the lens mask')        
        H_fx = QuadCurvMask(r=self.fx, plane='x').get_mask(dfl)        
        H_fy = QuadCurvMask(r=self.fy, plane='y').get_mask(dfl)
        self.mask = H_fx * H_fy
      
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

dfl1 = generate_dfl(**kwargs);  #Gaussian beam defenition
dfl2 = generate_dfl(**kwargs);  #Gaussian beam defenition

#plot_dfl(dfl, fig_name='before', phase=0)

#QuadCurvMask(r = 0, plane='x').apply(dfl1)
#H1 = QuadCurvMask(r = 25, plane='x').get_mask(dfl1)
#H2 = QuadCurvMask(r = 25, plane='y').get_mask(dfl1)
#print(H1)
#print('aaaaaaaaaaaaaa')
#print(H2)
l = ThinLens(fx=15, fy=10)

##
line = (l)
lat = OpticsLine(line)
###
dfl2 = propagate(lat, dfl2)                

#plot_dfl(dfl1, fig_name='after_dfl1', phase=0)
plot_dfl(dfl2, fig_name='after_dfl2', phase=0)



























