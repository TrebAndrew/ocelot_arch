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
    
    elif element.__class__ is FreeSpace:
        element.mask = DriftMask(element.l, element.mx, element.my)
    
  
### goes to optics_element.py
class FreeSpace(OpticsElement):
    """
    Drift element
    """
    def __init__(self, l=0., mx=1, my=1, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.l = l
        self.mx = mx
        self.my = my

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "
class QuadCurvMask(Mask):

    def __init__(self, r, plane):
        Mask.__init__(self)
        self.r = r #is the radius of curvature
        self.plane = plane #is the plane in which the curvature is applied
        self.mask = None #is the transfer function itself
        self.domains = 's' #is the domain in which the transfer function is calculated
        self.domain_z = None #is the domain in which wavefront curvature is introduced 

    def apply(self, dfl):
        
        domains = dfl.domain_z, dfl.domain_xy

        if self.domain_z is None:
            self.domain_z = dfl.domain_z
        
        _logger.debug('curving radiation wavefront by {}m in {} domain'.format(self.r, self.domain_z))
        
        if np.size(self.r) == 1:
            if self.r == 0:
                raise ValueError('radius of curvature should not be zero')
            elif self.r == np.inf:
                _logger.debug(ind_str + 'radius of curvature is infinite, skipping')
                return
            else:
                pass
            
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
            k = 2 * np.pi /dfl.scale_z() # <- change on somethenin' more reliable

            if np.size(self.r) == 1:
                self.mask = np.exp(-1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / self.r) #H = exp(-i * k / 2 * (x^2 + y^2))
            elif np.size(self.r) == dfl.Nz():
                self.mask = np.exp(-1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / self.r[:, np.newaxis, np.newaxis])
            else:
                raise ValueError('wrong dimensions of radius of curvature')    

        elif self.domain_z == 't':
            k = 2 * np.pi / dfl.xlamds

            if np.size(self.r) == 1:
                self.mask = np.exp(-1j * k / 2 * arg2 / self.r)
            elif np.size(self.r) == dfl.Nz():
                self.mask = np.exp(-1j * k / 2 * arg2[np.newaxis, :, :] / self.r[:, np.newaxis, np.newaxis])
            else:
                raise ValueError('wrong dimensions of radius of curvature')
        else:
            ValueError('domain_z should be in ["f", "t", None]')
            
        return self.mask
    
class PropMask(Mask):
    """
    Angular-spectrum propagation for fieldfile
    
    can handle wide spectrum
      (every slice in freq.domain is propagated
       according to its frequency)
    no kx**2+ky**2<<k0**2 limitation
    
    dfl is the RadiationField() object
    
    z0 is the propagation distance in [m]
    for 'ks' domain propagation
        no Fourier transform to frequency domain is done
        assumes no angular dispersion (true for plain FEL radiation)
        assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
    'kf' propagation is a default option
    
    z>0 -> forward direction
    z<0 -> backward direction
    z=0 return original
    """   
    def __init__(self, z0):
        Mask.__init__(self)
        self.z0 = z0 
        self.mask = None
        self.domains = 'kf'

    def apply(self, dfl):
        _logger.info('propagating dfl file by %.2f meters' % (self.z0))
        
        if self.z0 == 0:
            _logger.debug(ind_str + 'z0=0, returning original')
            return dfl

        start = time.time()

        domains = dfl.domains()

        if self.domains == 'kf' or self.domains == 'kf' or self.domains == 'k':
            dfl.to_domain(self.domains) #the field is transformed in inverce space domain and, optionaly, in 'f' or 't' domains
        else:
            raise ValueError("domains value should be 'kf' or 'kt' or 'k'")
            
        if self.mask is None:
            self.get_mask(dfl) # get H transfer function 

        dfl.fld *= self.mask # E = E_0 * H convolution in inverse space domain
        
        dfl.to_domain(domains) # back to original domain !!! 

        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        return dfl
        
    def get_mask(self, dfl):
     
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.xlamds = dfl.xlamds
        self.shape = dfl.shape()
        
        k_x, k_y = np.meshgrid(self.kx(), self.ky())
        
        if dfl.domain_z is 'f':
            k = self.kz()
            self.mask = [np.exp(1j * self.z0 * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())] #H = exp(iz0(k^2 - kx^2 - ky^2)^(1/2) - k)
        else:
            k = 2 * np.pi / dfl.xlamds
            self.mask = [np.exp(1j * self.z0 * (np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2) - k)) for i in range(dfl.Nz())]  #H = exp(iz0(k^2 - kx^2 - ky^2)^(1/2) - k)

        return self.mask

class Prop_mMask(Mask):
    """
    Angular-spectrum propagation for fieldfile
    
    can handle wide spectrum
      (every slice in freq.domain is propagated
       according to its frequency)
    no kx**2+ky**2<<k0**2 limitation
    
    dfl is the RadiationField() object
    
    z0 is the propagation distance in [m]
    m is the output mesh size in terms of input mesh size (m = L_out/L_inp)
    for 'ks' domain propagation
        no Fourier transform to frequency domain is done
        assumes no angular dispersion (true for plain FEL radiation)
        assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
    'kf' propagation is a default option
    
    z>0 -> forward direction
    z<0 -> backward direction
    z=0 return original
    """   
    def __init__(self, z0, mx, my):
        Mask.__init__(self)
        self.z0 = z0
        self.mx = mx
        self.my = my
        self.mask = None
        self.domains = 'kf'

    def apply(self, dfl):
        
        _logger.info('propagating dfl file by %.2f meters' % (self.z0))

        if self.z0 == 0:
            _logger.debug(ind_str + 'z0=0, returning original')
            return dfl
        
        start = time.time()
    
        if self.mx != 1:
#            dfl.curve_wavefront(-self.z0 / (1 - self.mx), plane='x')
            dfl = QuadCurvMask(r = -self.z0 / (1 - self.mx), plane='x').apply(dfl)
        if self.my != 1:
#            dfl.curve_wavefront(-self.z0 / (1 - self.my), plane='y')
            dfl = QuadCurvMask(r = -self.z0 / (1 - self.my), plane='y').apply(dfl)
        
        domains = dfl.domains()

        if self.domains == 'kf' or self.domains == 'kf' or self.domains == 'k':
            dfl.to_domain(self.domains) #the field is transformed in inverce space domain and, optionaly, in 'f' or 't' domains
        else:
            raise ValueError("domains value should be 'kf' or 'kt' or 'k'")
        
        if self.mask is None:
            self.get_mask(dfl) # get H transfer function 
            
        dfl.fld *= self.mask #E = E_0 * H convolution in inverse space domain
        
        dfl.dx *= self.mx #transform grid to output mesh size
        dfl.dy *= self.my        
        
        dfl.to_domain(domains) # back to original domain
        
        if self.mx != 1:
#            dfl.curve_wavefront(-self.mx / (self.mx - 1), plane='x')
            dfl = QuadCurvMask(r = -self.mx * self.z0 / (self.mx - 1), plane='x').apply(dfl)
        if self.my != 1:
#            dfl.curve_wavefront(-self.my / (self.my - 1), plane='y')
            dfl = QuadCurvMask(r = -self.my * self.z0 / (self.my - 1), plane='y').apply(dfl)

        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        return dfl
        
    def get_mask(self, dfl):
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.xlamds = dfl.xlamds
        self.shape = dfl.shape()
        
        k_x, k_y = np.meshgrid(self.kx(), self.ky())
        k = self.kz()

        if self.domains == 'kf':
            k = self.kz()
            Hx = [np.exp(1j * self.z0/self.mx * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i])) for i in range(dfl.Nz())][0] #Hx = exp(iz0/mx(k^2 - kx^2)^(1/2) - k)
            Hy = [np.exp(1j * self.z0/self.my * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())][0] #Hy = exp(iz0/my(k^2 - ky^2)^(1/2) - k)                  
            self.mask = Hx*Hy
        elif self.domains == 'ks':
            k = 2 * np.pi / dfl.xlamds
            Hx = [np.exp(1j * self.z0/self.mx * (np.sqrt(k ** 2 - k_x ** 2) - k)) for i in range(dfl.Nz())][0] 
            Hy = [np.exp(1j * self.z0/self.my * (np.sqrt(k ** 2 - k_y ** 2) - k)) for i in range(dfl.Nz())][0]          
            self.mask = Hx*Hy
        else: 
            raise ValueError('wrong field domain, domain must be ks or kf ')    
            
        return self.mask
    
class DriftMask(Mask):

    def __init__(self, z0, mx, my):
        Mask.__init__(self)
        self.z0 = z0
        self.mx = mx #
        self.my = my #
        self.mask = None
        self.type = 'PropMask'  #type of propagation. also may be 
        # 'Fraunhofer_Propagator'
        # 'Fresnel_Propagator' . . .
    def apply(self, dfl): 
        
        if self.mask is None:         
            if self.type == 'PropMask' and self.mx == 1 and self.my == 1:
                dfl = MaxwPropagator(z0 = self.z0).apply(dfl)
            elif self.type == 'Prop_mMask' and (self.mx != 1 or self.my != 1):
                dfl = MaxwPropagator_m(z0 = self.z0, mx = self.mx, my = self.my).apply(dfl)
            elif self.type == 'Fraunhofer_Propagator':
                pass
            elif self.type == 'Fresnel_Propagator':
                pass
            else:
                raise ValueError("check a propagation type")
            
        return dfl
        
    def get_mask(self, dfl):
        if self.type == 'PropMask' and self.mx == 1 and self.my == 1:
            self.mask = MaxwPropagator(z0 = self.z0).get(dfl)
        elif self.type == 'Prop_mMask' and self.mx != 1 and self.my != 1:
            self.mask = MaxwPropagator_m(z0 = self.z0, mx = self.mx, my = self.my).get(dfl)
        elif self.type == 'Fraunhofer_Propagator':
            pass
        elif self.type == 'Fresnel_Propagator':
            pass
        else:
            raise ValueError("check a propagation type")
     
        
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

#dfl = MaxwPropogator_m(z0=15, mx=1, my=1).apply(dfl)

dfl1 = deepcopy(dfl)
dfl2 = deepcopy(dfl)

dfl1 = PropMask(z0=100).apply(dfl1)
dfl2 = Prop_mMask(z0=100, mx=1.5, my=1.5).apply(dfl2)

#
#line = (RectApp, EllipsApp)
#lat = OpticsLine(line)
##
#dfl = propagate(lat, dfl)                

plot_dfl(dfl2, fig_name='after_dfl2', phase=0)
plot_dfl(dfl1, fig_name='after_dfl1', phase=0)



























