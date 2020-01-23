'''
simple optical system 
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


### goes to new_wave.py ### 
class Grid:
    
    def __init__(self, shape=(0, 0, 0)):
        self.dx = []
        self.dy = []
        self.dz = []
        self.shape = shape 
        self.xlamds = None #picky point!!!
    def Lz(self):  
        '''
        full longitudinal mesh size
        '''
        return self.dz * self.Nz()

    def Ly(self):  
        '''
        full transverse vertical mesh size
        '''
        return self.dy * self.Ny()

    def Lx(self):  
        '''
        full transverse horizontal mesh size
        '''
        return self.dx * self.Nx()

    def Nz(self):
        '''
        number of points in z
        '''
        return self.shape[0]

    def Ny(self):
        '''
        number of points in y
        '''
        return self.shape[1]

    def Nx(self):
        '''
        number of points in x
        '''
        return self.shape[2]
    
    def x(self):
        return np.linspace(-self.Lx() / 2, self.Lx() / 2, self.Nx())
    
    def kx(self):
        k = 2 * np.pi / self.dx
        return np.linspace(-k / 2, k / 2, self.Nx())
    
    def y(self):
        return np.linspace(-self.Ly() / 2, self.Ly() / 2, self.Ny())
    
    def ky(self):
        k = 2 * np.pi / self.dy
        return np.linspace(-k / 2, k / 2, self.Ny())

    def z(self):
        return np.linspace(0, self.Lz(), self.Nz())

    def kz(self):
        dk = 2 * np.pi / self.Lz()
        k = 2 * np.pi / self.xlamds #picky point!!!
        return np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz())
    
    
### goes to optics_line.py ### may be arrized and replaced by a line "from ocelot.rad.optics_line import *"

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

    if element.__class__ is None:
        raise ValueError('Optics element must belong to the OpticsElement class')
    
    elif element.__class__ is ApertureRect:
        element.mask = ApertureRectMask(element.lx, element.ly)
#        mask.lx = element.lx
#        mask.ly = element.ly
#        element.mask = mask
        
    elif element.__class__ is ApertureEllips:
        element.mask = ApertureEllipsMask(element.ax, element.ay)
#        mask.ax = element.ax
#        mask.ay = element.ay
#        element.mask = mask
#        
    elif element.__class__ is ImperfectMirror:
        element.mask = MirrorMask(element.height_profile, element.hrms, element.angle, element.plane, element.lx, element.ly)

    elif element.__class__ is ThinLens:
        element.mask = LensMask(element.fx, element.fy)
  
    elif element.__class__ is FreeSpace:
        element.mask = DriftMask(element.l, element.mx, element.my)

    elif element.__class__ is DispersiveSection:
        element.mask = PhaseDelayMask(element.coeff, element.E_ph0)
        

### goes to propagation.py ### may be arrized and replaced by a line "from ocelot.rad.propagation import * "

def propagate(optics_line, dfl, optimize=False, dump=False):
    
    if optimize: #not implemented
        estimate_masks(optics_line, dfl)
        combine_elements(optics_line)
            
    for element in optics_line.sequence:
        element.mask.apply(dfl)
    return dfl

def estimate_masks(optics_line, dfl):#not implemented
    for element in optics_line.sequence:
        element.mask.get_mask(dfl)

def combine_elements(optics_line):#not implemented
    return optics_line

### goes to optics_elements.py ### may be arrized and replaced by a line "from ocelot.rad.optics_elements import *"
        
class OpticsElement:
    def __init__(self, eid=None):
        self.eid = eid
        self.domain = "sf"

    def apply(self, dfl):
        pass 
     
class FreeSpace(OpticsElement):
    """
    Drift element
    """
    def __init__(self, l=0., mx=1, my=1, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.l = l
        self.mx = mx
        self.my = my

class ThinLens(OpticsElement):
    """
    Lens element
    """
    def __init__(self, fx=np.inf, fy=np.inf, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.fx = fx
        self.fy = fy
        
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
        
class DispersiveSection(OpticsElement):
    """
    Dispersive Section
    """
    
    def __init__(self, coeff=[0], E_ph0=None, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.coeff = coeff
        self.E_ph0 = E_ph0

#class OpticsMarker(OpticsElement):
#    """
#    Drift element
#    """
#    def __init__(self, eid=None):
#        OpticsElement.__init__(self, eid=eid)
#
#    def apply(self, dfl):
#        pass

class ImperfectMirror(OpticsElement):
    def __init__(self, height_profile=None, hrms=0, lx=np.inf, ly=np.inf, angle=np.pi * 2 / 180, plane='x', eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.height_profile = height_profile
        self.hrms = hrms
        self.lx = lx
        self.ly = ly
        self.angle=angle
        self.plane=plane
        
#class HeightErrorProfile():
#    """
#    Drift element
#    """
#    def __init__(self, hrms=0, lx=1., ly=1., nx=1000, ny=1000, k_cutoff=0., psd=None, eid=None):
#        self.eid = eid
#        self.hrms = hrms
#        self.lx = lx
#        self.ly = ly
#        self.nx = nx
#        self.ny = ny
#        self.k_cutoff = k_cutoff
#        self.psd = psd

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "
        
class Mask(Grid):
    
    def __init__(self, shape=(0, 0, 0)):
        Grid.__init__(self, shape=shape)

    def apply(self, dfl):
        return dfl

    def get_mask(self, dfl):
        pass

    def __mul__(self, other):
        m = deepcopy(self)
        if other.__class__ in [self] and self.mask is not None and other.mask is not None:
            m.mask = self.mask * other.mask
            return m

class ApertureRectMask(Mask):
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

class ApertureEllipsMask(Mask):   
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
        self.mx = mx 
        self.my = my 
        self.mask = None
        self.type = 'PropMask'  #type of propagation, also may be 
        # 'Fraunhofer_Propagator'
        # 'Fresnel_Propagator' . . .
    def apply(self, dfl): 
        
        if self.mask is None:         
            if self.type == 'PropMask' and self.mx == 1 and self.my == 1:
                dfl = PropMask(z0 = self.z0).apply(dfl)
            
            elif self.type == 'Prop_mMask' or (self.mx != 1 or self.my != 1):
                dfl = Prop_mMask(z0 = self.z0, mx = self.mx, my = self.my).apply(dfl)
            
            elif self.type == 'Fraunhofer_Propagator':
                pass
            
            elif self.type == 'Fresnel_Propagator':
                pass
            
            else:
                raise ValueError("check a propagation type")
            
        return dfl
        
    def get_mask(self, dfl):
        if self.type == 'PropMask' and self.mx == 1 and self.my == 1:
            self.mask = PropMask(z0 = self.z0).get(dfl)
        
        elif self.type == 'Prop_mMask' and self.mx != 1 and self.my != 1:
            self.mask = Prop_mMask(z0 = self.z0, mx = self.mx, my = self.my).get(dfl)
        
        elif self.type == 'Fraunhofer_Propagator':
            pass
        
        elif self.type == 'Fresnel_Propagator':
            pass
        
        else:
            raise ValueError("check a propagation type")
     
        
        return self.mask

        
class PhaseDelayMask(Mask):
    """
    The function adds a phase shift to a fld object. The expression for the phase see in the calc_phase_delay function
    dfl   --- is a fld object
    coeff --- 
        coeff[0] =: measured in [rad]      --- phase
        coeff[1] =: measured in [fm s ^ 1] --- group delay
        coeff[2] =: measured in [fm s ^ 2] --- group delay dispersion (GDD)
        coeff[3] =: measured in [fm s ^ 3] --- third-order dispersion (TOD)
        ...
    E_ph0 --- energy with respect to which the phase shift is calculated
    """
    def __init__(self, coeff, E_ph0):
        Mask.__init__(self)
        self.coeff = coeff
        self.E_ph0 = E_ph0
        self.mask = None
        
    def apply(self, dfl):
        
        _logger.info('apply the frequency chirp')
        domains = dfl.domains()
        
        if self.mask is None: 
            self.get_mask(dfl)
        
        dfl.to_domain('f')
        dfl.fld *= self.mask
        dfl.to_domain(domains)
        
        _logger.info('done')
        return dfl

    def get_mask(self, dfl):
        """
        expression for the phase -- coeff[0] + coeff[1]*(w - w0)/1! + coeff[2]*(w - w0)**2/2! + coeff[3]*(w - w0)**3/3!
        coeff is a list with
        coeff[0] =: measured in [rad]      --- phase
        coeff[1] =: measured in [fm s ^ 1] --- group delay
        coeff[2] =: measured in [fm s ^ 2] --- group delay dispersion (GDD)
        coeff[3] =: measured in [fm s ^ 3] --- third-order dispersion (TOD)
        ...
        """
        _logger.info('get the frequency chirp mask')
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.shape = dfl.shape()
        self.xlamds = dfl.xlamds
        
        if self.E_ph0 == None:
            w0 = 2 * np.pi * speed_of_light / dfl.xlamds
    
        elif self.E_ph0 == 'center':
            _, lamds = np.mean([dfl.int_z(), dfl.scale_z()], axis=(1))
            w0 = 2 * np.pi * speed_of_light / lamds
    
        elif isinstance(self.E_ph0, str) is not True:
            w0 = 2 * np.pi * self.E_ph0 / h_eV_s
    
        else:
            raise ValueError("E_ph0 must be None or 'center' or some value")

        w = self.kz() * speed_of_light
        delta_w = w - w0

        _logger.debug('calculating phase delay')
        _logger.debug(ind_str + 'coeffs for compression = {}'.format(self.coeff))

        coeff_norm = [ci / (1e15) ** i / factorial(i) for i, ci in enumerate(self.coeff)]
        coeff_norm = list(coeff_norm)[::-1]
        _logger.debug(ind_str + 'coeffs_norm = {}'.format(coeff_norm))
        delta_phi = np.polyval(coeff_norm, delta_w)

        _logger.debug(ind_str + 'delta_phi[0] = {}'.format(delta_phi[0]))
        _logger.debug(ind_str + 'delta_phi[-1] = {}'.format(delta_phi[-1]))

        self.mask = np.exp(-1j * delta_phi)[:, np.newaxis, np.newaxis]

        _logger.debug(ind_str + 'done')
        
        return self.mask
        #####################################

class MirrorMask(Mask):
    """
    Class for simulating HeightProfile of highly polished mirror surface
    """
    def __init__(self, height_profile=None, hrms=0, angle=2*np.pi/180, plane='x', lx=np.inf, ly=np.inf, eid=None, shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.height_profile = height_profile 
        self.hrms = hrms #must have a size equals 2 
        self.angle = angle
        self.mask = None
        self.eid = eid
        
        if plane is 'x':
            self.lx = lx/np.sin(self.angle)
            self.ly = ly
        elif plane is 'y':
            self.lx = lx
            self.ly = ly/np.sin(self.angle)
        else:
            raise ValueError(" 'plane' must be 'x' or 'y' ")

    def apply(self, dfl):
        
        _logger.info('apply HeightProfile errors')
        start = time.time()

        if self.mask is None: 
            self.get_mask(dfl)

        dfl.fld *= self.mask
        
        t_func = time.time() - start
        _logger.debug(ind_str + 'done in {}'.format(t_func))


    def get_mask(self, dfl):
        _logger.info('getting HeightProfile errors')
        
        if self.mask is None:
            heightErr_x = HeightErrorMask_1D(height_profile=self.height_profile, hrms=self.hrms, axis='x', angle=self.angle)
            heightErr_x.get_mask(dfl) 
            heightErr_y = HeightErrorMask_1D(height_profile=self.height_profile, hrms=self.hrms, axis='y', angle=self.angle)
            heightErr_y.get_mask(dfl) 
            RectApp = RectMask(lx=self.lx, ly=self.ly)
            RectApp.get_mask(dfl)
            self.mask = heightErr_x.mask * heightErr_y.mask * RectApp.mask
        return self.mask
        
        
class HeightErrorMask_1D(Mask):
    """
    Mask for simulating HeightProfile of highly polished mirror surface

    :param hrms: [meters] height errors root mean square
    :param length: [meters] length of the surface
    :param points_number: number of points (pixels) at the surface
    :param wavevector_cutoff: [1/meters] point on k axis for cut off small wavevectors (large wave lengths) in the PSD
                                    (with default value 0 effects on nothing)
    :param psd: [meters^3] 1d array; power spectral density of surface (if not specified, will be generated)
            (if specified, must have shape = (points_number // 2 + 1, ), otherwise it will be cut to appropriate shape)
    :param seed: seed for np.random.seed() to allow reproducibility
    :return: HeightProfile object
    """

    def __init__(self, height_profile=None, hrms=0, axis='x', angle=np.pi * 2 / 180, seed=None):
        Mask.__init__(self)
        self.mask = None
        self.height_profile = height_profile 
        self.hrms = hrms
        self.axis = axis
        self.angle = angle
        self.seed = seed
        
    def apply(self, dfl):
        
        _logger.info('apply HeightProfile errors')
        start = time.time()

        if self.mask is None: 
            self.get_mask(dfl)

        dfl.fld *= self.mask
        
        t_func = time.time() - start
        _logger.debug(ind_str + 'done in {}'.format(t_func))

        return dfl
        
    def get_mask(self, dfl):
        
        self.dx = dfl.dx
        self.dy = dfl.dy
        self.dz = dfl.dz
        self.shape = dfl.shape()
        
        dict_axes = {'z': 0, 'y': 1, 'x': 2}
        dl = {0: dfl.dz, 1: dfl.dy, 2: dfl.dx}
        
        if isinstance(self.axis, str):
            axis = dict_axes[self.axis]
 
        n = dfl.fld.shape[axis]
        eff_l = n * dl[axis] / np.sin(self.angle)
        if self.height_profile is None:
            if self.hrms is None:
                _logger.error('hrms and height_profile not specified')
                raise ValueError('hrms and height_profile not specified')
            self.height_profile = HeightProfile()
            self.height_profile = self.height_profile.generate_1d_profile(self.hrms, L=eff_l, N=n, seed=self.seed)
        
        elif self.height_profile.L != eff_l or self.height_profile.N != n:
            if self.height_profile.L < eff_l:
                _logger.warning(
                    'provided height profile length {} is smaller than projected footprint {}'.format(height_profile.L,
                                                                                                      eff_l))  # warn and pad with zeroes
    
            # interpolation of height_profile to appropriate sizes
            _logger.info(ind_str + 'given height_profile was interpolated to length and '.format(dfl.shape()))
            s = np.linspace(-eff_l / 2, eff_l / 2, n)
            h = np.interp(s, height_profile.s, height_profile.h, right=height_profile.h[0], left=height_profile.h[-1])
            self.height_profile = HeightProfile()
            self.height_profile.points_number = dfl.fld.shape[axis]
            self.height_profile.L = eff_l
            self.height_profile.s = s
            self.height_profile.h = h
    
        phase_delay = 2 * 2 * np.pi * np.sin(self.angle) * self.height_profile.h / dfl.xlamds
        
        if self.axis == 'x':
            self.mask = np.exp(1j * phase_delay)[np.newaxis, np.newaxis, :]
        elif self.axis == 'y':
            self.mask = np.exp(1j * phase_delay)[np.newaxis, :, np.newaxis]
        elif self.axis == 'z':
            self.mask = np.exp(1j * phase_delay)[:, np.newaxis, np.newaxis]             
        return self.mask
    
    
### script itself ###
        
#optics elements check
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

d = FreeSpace(l=200, mx=1.5, my=1.5)
l = ThinLens(fx=25, fy=25)

appEl = ApertureEllips(ax=0.001, ay=0.001, cx=0, cy=0)
appRect =  ApertureRect(lx=0.001, ly=0.001, cx=0, cy=0)
#
line = (d)#, l, app)

lat = OpticsLine(line)

dfl = propagate(lat, dfl)

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
plot_dfl(dfl, fig_name='after1', phase=1)
#




#%%
### phase delay check ###

coeff = [0,0,-250,100]

seed_dfl = generate_dfl(1239.8/500*1e-9, shape=(3,3,2001), dgrid=(1e-3,1e-3,400e-6), power_rms=(0.5e-3,0.5e-3,0.5e-6), 
                        power_center=(0,0,None), power_angle=(0,0), power_waistpos=(0,0), #wavelength=[4.20e-9,4.08e-9], 
                        zsep=None, freq_chirp=0, energy=None, power=10e6, debug=1)

SASE_dfl = imitate_sase_dfl(1239.8/500*1e-9,rho=4e-4, shape=(3,3,2001), dgrid=(1e-3,1e-3,400e-6), power_rms=(0.5e-3,0.5e-3,5e-6), 
                        power_center=(0,0,None), power_angle=(0,0), power_waistpos=(0,0),
                        zsep=None, energy=None, power=10e6, debug=1)

#plot_dfl(seed_dfl, fig_name='before_seed', phase=0)
#plot_dfl(SASE_dfl, fig_name='before_SASE', phase=0)

#wig = wigner_dfl(seed_dfl)
SASE_wig = wigner_dfl(SASE_dfl)

#plot_wigner(wig, fig_name = 'before', plot_moments=0)
plot_wigner(SASE_wig, fig_name = 'SASE_before', plot_moments=0)

#plot_wigner(wig, fig_name = 'before', plot_moments=0)
#plot_wigner(SASE_wig, fig_name = 'SASE_before', plot_moments=0)

#PhaseDelayMask(coeff = coeff).apply(seed_dfl)
#PhaseDelayMask(coeff = coeff).apply(SASE_dfl)

PhaseDelay = DispersiveSection(coeff)
line = (PhaseDelay)#, l, app)
lat = OpticsLine(line)

SASE_dfl = propagate(lat, SASE_dfl)

#wig = wigner_dfl(seed_dfl)
SASE_wig = wigner_dfl(SASE_dfl)

#plot_wigner(wig, fig_name = 'after', plot_moments=0)
plot_wigner(SASE_wig, fig_name = 'SASE_after', plot_moments=0)

#plot_dfl(seed_dfl, fig_name='after_seed', phase=0)
#plot_dfl(SASE_dfl, fig_name='after_SASE', phase=0)

#%%







