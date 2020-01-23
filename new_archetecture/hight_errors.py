'''
hight_errors.py for pre pull request check v0.1
no apperture for the mirror is implemented
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


###goes to wave.py
class HeightProfile:
    """
    1d surface of mirror
    """

    def __init__(self):
        self.N = None #points number
        self.L = None #length of the surface
        self.h = None #height profile
        self.s = None #np.linspace(-L / 2, L / 2, N)

    def hrms(self):
        return np.sqrt(np.mean(np.square(self.h)))

    def set_hrms(self, rms):
        self.h *= rms / self.hrms()

    def psd(self):
        psd = 1 / (self.L * np.pi) * np.square(np.abs(np.fft.fft(self.h) * self.L / self.N))
        psd = psd[:len(psd) // 2]
        k = np.pi / self.L * np.linspace(0, self.N, self.N // 2)
        # k = k[len(k) // 2:]
        return (k, psd)
    
    def generate_1d_profile(self, hrms, L=0.1, N=1000, k_cut=0, psd=None, seed=None):
        """
        Method for generating HeightProfile of highly polished mirror surface
    
        :param hrms: [meters] height errors root mean square
        :param L: [meters] length of the surface
        :param N: number of points (pixels) at the surface
        :param k_cut: [1/meters] point on k axis for cut off small wavevectors (large wave lengths) in the PSD
                                        (with default value 0 effects on nothing)
        :param psd: [meters^3] 1d array; power spectral density of surface (if not specified, will be generated)
                (if specified, must have shape = (points_number // 2 + 1, ), otherwise it will be cut to appropriate shape)
        :param seed: seed for np.random.seed() to allow reproducibility
        :return: HeightProfile object
        """
    
        _logger.info('generating 1d surface with rms: {} m; and shape: {}'.format(hrms, (N,)))
        _logger.warning(ind_str + 'in beta')
    
        # getting the heights map
        if seed is not None:
            np.random.seed(seed)
            
        if psd is None:
            k = np.pi / L * np.linspace(0, N, N // 2 + 1)
            # defining linear function PSD(k) in loglog plane
            a = -2  # free term of PSD(k) in loglog plane
            b = -2  # slope of PSD(k) in loglog plane
            psd = np.exp(a * np.log(10)) * np.exp(b * np.log(k[1:]))
            psd = np.append(psd[0], psd)  # ??? It doesn*t important, but we need to add that for correct amount of points
            if k_cut != 0:
                idx = find_nearest_idx(k, wavevector_cutoff)
                psd = np.concatenate((np.full(idx, psd[idx]), psd[idx:]))
        elif psd.shape[0] > N // 2 + 1:
            psd = psd[:N // 2 + 1]
    
        phases = np.random.rand(N // 2 + 1)
        height_profile = HeightProfile()
        height_profile.N = N
        height_profile.L = L
        height_profile.s = np.linspace(-L / 2, L / 2, N)
        height_profile.h = (N / L) * np.fft.irfft(np.sqrt(L * psd) * np.exp(1j * phases * 2 * np.pi),
                                                                   n=N) / np.sqrt(np.pi)
        # scaling height_map
        height_profile.set_hrms(hrms)
        
        np.random.seed()
        
        return height_profile
    

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
            print(element, element.angle)

            get_transfer_function(element)

    def estimate_mesh(self):
        for element in self.sequence:
            element.mesh = 0
            
def get_transfer_function(element):
#    element.mask = Mask()
    #better to implement via 'try'?
    if element.__class__ is None:
        raise ValueError('Optics element must belong to the OpticsElement class')
    
    elif element.__class__ is ImperfectMirror:
        element.mask = MirrorMask(element.height_profile, element.hrms, element.angle, element.plane, element.lx, element.ly)

  
### goes to optics_element.py

class ImperfectMirror(OpticsElement):
    def __init__(self, height_profile=None, hrms=0, lx=np.inf, ly=np.inf, angle=np.pi * 2 / 180, plane='x', eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.height_profile = height_profile
        self.hrms = hrms
        self.lx = lx
        self.ly = ly
        self.angle=angle
        self.plane=plane
#        print(self.angle, 'ImperfectMirror')

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "

class RectMask(Mask): #must be here for mirror mask
    def __init__(self, lx=np.inf, ly=np.inf, shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.lx = lx
        self.ly = ly
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
    


### main body
    
#optics elements check
#dfl = RadiationField()
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
#plot_dfl(dfl, fig_name='before', phase=0)

#height_profile = HeightProfile().generate_1d_profile(hrms=1e-9)        
#mirror = ImperfectMirror(height_profile=height_profile)

mirror = ImperfectMirror(hrms=1e-9, lx=0.0007, ly=0.0007, angle=15*np.pi/180, plane='x')

line = (mirror)
lat = OpticsLine(line)

#hightErr = MirrorMask(hrms=1e-9)#HightErrorMask_1D(hrms=1e-9)
#hightErr.apply(dfl)

dfl = propagate(lat, dfl)                
plot_dfl(dfl, fig_name='after', phase=0)



























