from numpy import random
from numpy.linalg import norm
import numpy as np
from math import factorial
from numpy import inf, complex128, complex64
import scipy
import numpy.fft as fft
from copy import deepcopy
import time
import os

# from ocelot.optics.elements import *
from ocelot.common.globals import *
from ocelot.common.math_op import find_nearest_idx, fwhm, std_moment, bin_scale, bin_array, mut_coh_func
from ocelot.common.py_func import filename_from_path
from ocelot.gui.dfl_plot import plot_dfl
from ocelot.optics.wave import RadiationField

# from ocelot.optics.utils import calc_ph_sp_dens
# from ocelot.adaptors.genesis import *
# import ocelot.adaptors.genesis as genesis_ad
# GenesisOutput = genesis_ad.GenesisOutput
from ocelot.common.ocelog import *
_logger = logging.getLogger(__name__)

import multiprocessing
nthread = multiprocessing.cpu_count()

try:
    import pyfftw
    fftw_avail = True
except ImportError:
    print("wave.py: module PYFFTW is not installed. Install it if you want speed up dfl wavefront calculations")
    fftw_avail = False

__author__ = "Svitozar Serkez, Andrei Trebushinin, Mykola Veremchuk"
### just must to be here for generating dfl :_)

def generate_gaussian_dfl(xlamds=1e-9, shape=(51, 51, 100), dgrid=(1e-3, 1e-3, 50e-6), power_rms=(0.1e-3, 0.1e-3, 5e-6),
                          power_center=(0, 0, None), power_angle=(0, 0), power_waistpos=(0, 0), wavelength=None,
                          zsep=None, freq_chirp=0, en_pulse=None, power=1e6, **kwargs):
    """
    generates RadiationField object
    narrow-bandwidth, paraxial approximations

    xlamds [m] - central wavelength
    shape (x,y,z) - shape of field matrix (reversed) to dfl.fld
    dgrid (x,y,z) [m] - size of field matrix
    power_rms (x,y,z) [m] - rms size of the radiation distribution (gaussian)
    power_center (x,y,z) [m] - position of the radiation distribution
    power_angle (x,y) [rad] - angle of further radiation propagation
    power_waistpos (x,y) [m] downstrean location of the waist of the beam
    wavelength [m] - central frequency of the radiation, if different from xlamds
    zsep (integer) - distance between slices in z as zsep*xlamds
    freq_chirp dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
    en_pulse, power = total energy or max power of the pulse, use only one
    """

    start = time.time()

    if dgrid[2] is not None and zsep is not None:
        if shape[2] == None:
            shape = (shape[0], shape[1], int(dgrid[2] / xlamds / zsep))
        else:
            _logger.error(ind_str + 'dgrid[2] or zsep should be None, since either determines longiduninal grid size')

    _logger.info('generating radiation field of shape (nz,ny,nx): ' + str(shape))
    if 'energy' in kwargs:
        _logger.warn(ind_str + 'rename energy to en_pulse, soon arg energy will be deprecated')
        en_pulse = kwargs.pop('energy', 1)

    dfl = RadiationField((shape[2], shape[1], shape[0]))

    k = 2 * np.pi / xlamds

    dfl.xlamds = xlamds
    dfl.domain_z = 't'
    dfl.domain_xy = 's'
    dfl.dx = dgrid[0] / dfl.Nx()
    dfl.dy = dgrid[1] / dfl.Ny()

    if dgrid[2] is not None:
        dz = dgrid[2] / dfl.Nz()
        zsep = int(dz / xlamds)
        if zsep == 0:
            _logger.warning(
                ind_str + 'dgrid[2]/dfl.Nz() = dz = {}, which is smaller than xlamds = {}. zsep set to 1'.format(dz,
                                                                                                                 xlamds))
            zsep = 1
        dfl.dz = xlamds * zsep
    elif zsep is not None:
        dfl.dz = xlamds * zsep
    else:
        _logger.error('dgrid[2] or zsep should be not None, since they determine longiduninal grid size')

    rms_x, rms_y, rms_z = power_rms  # intensity rms [m]
    _logger.debug(ind_str + 'rms sizes = [{}, {}, {}]m (x,y,z)'.format(rms_x, rms_y, rms_z))
    xp, yp = power_angle
    x0, y0, z0 = power_center
    zx, zy = power_waistpos

    if z0 == None:
        z0 = dfl.Lz() / 2

    xl = np.linspace(-dfl.Lx() / 2, dfl.Lx() / 2, dfl.Nx())
    yl = np.linspace(-dfl.Ly() / 2, dfl.Ly() / 2, dfl.Ny())
    zl = np.linspace(0, dfl.Lz(), dfl.Nz())
    z, y, x = np.meshgrid(zl, yl, xl, indexing='ij')

    qx = 1j * np.pi * (2 * rms_x) ** 2 / xlamds + zx
    qy = 1j * np.pi * (2 * rms_y) ** 2 / xlamds + zy
    qz = 1j * np.pi * (2 * rms_z) ** 2 / xlamds

    if wavelength.__class__ in [list, tuple, np.ndarray] and len(wavelength) == 2:
        domega = 2 * np.pi * speed_of_light * (1 / wavelength[0] - 1 / wavelength[1])
        dt = (z[-1, 0, 0] - z[0, 0, 0]) / speed_of_light
        freq_chirp = domega / dt / 1e30 / zsep
        # freq_chirp = (wavelength[1] - wavelength[0]) / (z[-1,0,0] - z[0,0,0])
        _logger.debug(ind_str + 'difference wavelengths {} {}'.format(wavelength[0], wavelength[1]))
        _logger.debug(ind_str + 'difference z {} {}'.format(z[-1, 0, 0], z[0, 0, 0]))
        _logger.debug(ind_str + 'd omega {}'.format(domega))
        _logger.debug(ind_str + 'd t     {}'.format(dt))
        _logger.debug(ind_str + 'calculated chirp {}'.format(freq_chirp))
        wavelength = np.mean([wavelength[0], wavelength[1]])

    if wavelength == None and xp == 0 and yp == 0:
        phase_chirp_lin = 0
    elif wavelength == None:
        phase_chirp_lin = x * np.sin(xp) + y * np.sin(yp)
    else:
        phase_chirp_lin = (z - z0) / dfl.dz * (dfl.xlamds - wavelength) / wavelength * xlamds * zsep + x * np.sin(
            xp) + y * np.sin(yp)

    if freq_chirp == 0:
        phase_chirp_quad = 0
    else:
        # print(dfl.scale_z() / speed_of_light * 1e15)
        # phase_chirp_quad = freq_chirp *((z-z0)/dfl.dz*zsep)**2 * xlamds / 2# / pi**2
        phase_chirp_quad = freq_chirp / (speed_of_light * 1e-15) ** 2 * (zl - z0) ** 2 * dfl.xlamds  # / pi**2
        # print(phase_chirp_quad.shape)

    # if qz == 0 or qz == None:
    #     dfl.fld = np.exp(-1j * k * ( (x-x0)**2/2/qx + (y-y0)**2/2/qy - phase_chirp_lin + phase_chirp_quad ) )
    # else:
    arg = np.zeros_like(z).astype('complex128')
    if qx != 0:
        arg += (x - x0) ** 2 / 2 / qx
    if qy != 0:
        arg += (y - y0) ** 2 / 2 / qy
    if abs(qz) == 0:
        idx = abs(zl - z0).argmin()
        zz = -1j * np.ones_like(arg)
        zz[idx, :, :] = 0
        arg += zz
    else:
        arg += (z - z0) ** 2 / 2 / qz
        # print(zz[:,25,25])

    if np.size(phase_chirp_lin) > 1:
        arg -= phase_chirp_lin
    if np.size(phase_chirp_quad) > 1:
        arg += phase_chirp_quad[:, np.newaxis, np.newaxis]
    dfl.fld = np.exp(-1j * k * arg)  # - (grid[0]-z0)**2/qz
    # dfl.fld = np.exp(-1j * k * ( (x-x0)**2/2/qx + (y-y0)**2/2/qy + (z-z0)**2/2/qz - phase_chirp_lin + phase_chirp_quad) ) #  - (grid[0]-z0)**2/qz

    if en_pulse != None and power == None:
        dfl.fld *= np.sqrt(en_pulse / dfl.E())
    elif en_pulse == None and power != None:
        dfl.fld *= np.sqrt(power / np.amax(dfl.int_z()))
    else:
        _logger.error('Either en_pulse or power should be defined')
        raise ValueError('Either en_pulse or power should be defined')

    dfl.filePath = ''

    t_func = time.time() - start
    _logger.debug(ind_str + 'done in %.2f sec' % (t_func))

    return dfl

# here the new arch begins
### goes to new_wave.py ###
class Grid:
    """
    Grid object defines spatial-frequency 3D mesh for Radiation Field class and Mask class    
    :params (dx, dy, dz): spatial step of the mesh 
    :param shape: number of point in each direction of 3D data mesh
    :param xlamds: carrier wavelength of the wavepacket
    :param used_aprox: the approximation used in solving the wave equation. 
    OCELOT works in Slowly Varying Amplitude Approximation (SVAA)
    """
    def __init__(self, shape=(0,0,0)):
        self.dx = []
        self.dy = []
        self.dz = []
        self.shape = shape
        self.xlamds = None
        self.used_aprox = 'SVAA'

    def copy_grid(self, other, version=2):
        """
        TODO
        write documentation
        """
        if version == 1:
            self.dx = other.dx
            self.dy = other.dy
            self.dz = other.dz
            self.shape = other.shape
            
            self.xlamds = other.xlamds
            self.used_aprox = other.used_aprox
            
        elif version == 2: #I(Andrei) had an idea of setting attributes with the same memory
# address for Mask and RadiationField objects, to synchronize these object attributes "automatically"
            attr_list = np.intersect1d(dir(self),dir(other))
            for attr in attr_list:
                if attr.startswith('__') or callable(getattr(self, attr)):
                    continue
                setattr(self, attr, getattr(other, attr))
        else:
            raise ValueError
            
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
    
    def grid_x(self):
        return np.linspace(-self.Lx() / 2, self.Lx() / 2, self.Nx())
    
    def grid_kx(self):
        k = 2 * np.pi / self.dx
        return np.linspace(-k / 2, k / 2, self.Nx())
    
    def grid_y(self):
        return np.linspace(-self.Ly() / 2, self.Ly() / 2, self.Ny())
    
    def grid_ky(self):
        k = 2 * np.pi / self.dy
        return np.linspace(-k / 2, k / 2, self.Ny())
          
    def grid_z(self):
        return np.linspace(0, self.Lz(), self.Nz())
    
    def grid_kz(self):
        dk = 2 * pi / self.Lz()
        k = 2 * pi / self.xlamds       
        return np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz())    
    
#    def grid_t(self):
#        return grid_z()/speed_of_light
#    
#    def grid_f(self):
#        return grid_kz/speed_of_light
#           

class RadiationField(Grid):
    """
    3d or 2d coherent radiation distribution, *.fld variable is the same as Genesis dfl structure
    """

    def __init__(self, shape=(0,0,0)):
        Grid.__init__(self, shape=shape)
        self.fld = np.zeros(shape, dtype=complex128)  # (z,y,x)
        self.xlamds = None    # carrier wavelength [m]
        self.domain_z = 't'   # longitudinal domain (t - time, f - frequency)
        self.domain_x = 's'   # transverse domain (s - space, k - inverse space)
        self.domain_y = 's'   # transverse domain (s - space, k - inverse space)
        self.domain_xy = 's'  # transverse domain (s - space, k - inverse space)
        self.filePath = ''
                      
    def fileName(self):
        return filename_from_path(self.filePath)

    def copy_param(self, dfl1, version=1):
        if version == 1:
            self.dx = dfl1.dx
            self.dy = dfl1.dy
            self.dz = dfl1.dz
            self.xlamds = dfl1.xlamds
            self.domain_z = dfl1.domain_z
            self.domain_x = dfl1.domain_x
            self.domain_y = dfl1.domain_y

            self.domain_xy = dfl1.domain_xy
            self.filePath = dfl1.filePath
        elif version == 2: #does it link the address of these two objects only? : _) then exactly what we need for grid copying
            attr_list = dir(dfl1)
            for attr in attr_list:
                if attr.startswith('__') or callable(getattr(self, attr)):
                    continue
                if attr == 'fld':
                    continue
                setattr(self, attr, getattr(dfl1, attr))

    def __getitem__(self, i):
        return self.fld[i]

    def __setitem__(self, i, fld):
        self.fld[i] = fld

    def shape(self):
        '''
        returns the shape of fld attribute
        '''
        return self.fld.shape

    def domains(self):
        '''
        returns domains of the radiation field
        '''
        return self.domain_z, self.domain_xy

    def intensity(self):
        '''
        3d intensity, abs(fld)**2
        '''
        return self.fld.real ** 2 + self.fld.imag ** 2 # calculates faster

    def int_z(self):
        '''
        intensity projection on z
        power [W] or spectral density [arb.units]
        '''
        return np.sum(self.intensity(), axis=(1, 2))

    def ang_z_onaxis(self):
        '''
        on-axis phase
        '''
        xn = int((self.Nx() + 1) / 2)
        yn = int((self.Ny() + 1) / 2)
        fld = self[:, yn, xn]
        return np.angle(fld)

    def int_y(self):
        '''
        intensity projection on y
        '''
        return np.sum(self.intensity(), axis=(0, 2))

    def int_x(self):
        '''
        intensity projection on x
        '''
        return np.sum(self.intensity(), axis=(0, 1))

    def int_xy(self):
        # return np.swapaxes(np.sum(self.intensity(), axis=0), 1, 0)
        return np.sum(self.intensity(), axis=0)

    def int_zx(self):
        return np.sum(self.intensity(), axis=1)

    def int_zy(self):
        return np.sum(self.intensity(), axis=2)

    def E(self):
        '''
        energy in the pulse [J]
        '''
        if self.Nz() > 1:
            return np.sum(self.intensity()) * self.Lz() / self.Nz() / speed_of_light
        else:
            return np.sum(self.intensity())

    def scale_kx(self):  # scale in meters or meters**-1
        _logger.warning('"scale_kx" will be deprecated, use "grid_x and grid_kx" instead')

#        if 's' in [self.domain_xy, self.domain_x]:    # space domain
        if self.domain_xy == 's':    # space domain
            return self.grid_x()        

#        elif 'k' in [self.domain_xy, self.domain_x]:  # inverse space domain
        elif self.domain_xy == 'k':  # inverse space domain
            return self.grid_kx()
       
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_ky(self):  # scale in meters or meters**-1
        _logger.warning('"scale_ky" will be deprecated, use "grid_y and grid_ky" instead')
#        if 's' in [self.domain_xy, self.domain_y]:    # space domain
        if self.domain_xy == 's':    # space domain
            return self.grid_y()
#        elif 'k' in [self.domain_xy, self.domain_y]:  # inverse space domain
        if self.domain_xy == 'k':    # space domain
            return self.grid_ky()
        else:
            raise AttributeError('Wrong domain_xy attribute')
    
    def scale_kz(self):  # scale in meters or meters**-1
        _logger.warning('"scale_kz" will be deprecated, use "grid_z and grid_kz" instead')        
        if self.domain_z == 't':  # time domain
            return self.grid_z()
        elif self.domain_z == 'f':  # frequency domain
            return self.grid_kz()
        else:
            raise AttributeError('Wrong domain_z attribute')

    def scale_x(self):  # scale in meters or radians
#        _logger.warning('"scale_x" will be deprecated, use "grid_x and grid_kx" instead')        
        if self.domain_xy == 's':  # space domain
            return self.scale_kx()
        elif self.domain_xy == 'k':  # inverse space domain
            return self.scale_kx() * self.xlamds / 2 / np.pi
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_y(self):  # scale in meters or radians
        if self.domain_xy == 's':  # space domain
            return self.scale_ky()
        elif self.domain_xy == 'k':  # inverse space domain
            return self.scale_ky() * self.xlamds / 2 / np.pi
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_z(self):  # scale in meters
        if self.domain_z == 't':  # time domain
            return self.scale_kz()
        elif self.domain_z == 'f':  # frequency domain
            return 2 * pi / self.scale_kz()
        else:
            raise AttributeError('Wrong domain_z attribute')

    def ph_sp_dens(self):
        if self.domain_z == 't':
            dfl = deepcopy(self)
            dfl.fft_z()
        else:
            dfl = self
        pulse_energy = dfl.E()
        spec0 = dfl.int_z()
        freq_ev = h_eV_s * speed_of_light / dfl.scale_z()
        freq_ev_mean = np.sum(freq_ev * spec0) / np.sum(spec0)
        n_photons = pulse_energy / q_e / freq_ev_mean
        spec = calc_ph_sp_dens(spec0, freq_ev, n_photons)
        return freq_ev, spec

    def to_domain(self, domains='ts', **kwargs):
        """
        tranfers radiation to specified domains
        *domains is a string with one or two letters:
            ("t" or "f") and ("s" or "k")
        where
            't' (time); 'f' (frequency); 's' (space); 'k' (inverse space);
        e.g.
            't'; 'f'; 's'; 'k'; 'ts'; 'fs'; 'tk'; 'fk'
        order does not matter

        **kwargs are passed down to self.fft_z and self.fft_xy
        """
        _logger.debug('transforming radiation field to {} domain'.format(str(domains)))
        dfldomain_check(domains)

        for domain in domains:
            domain_o_z, domain_o_xy = self.domain_z, self.domain_xy
            if domain in ['t', 'f'] and domain is not domain_o_z:
                self.fft_z(**kwargs)
            if domain in ['s', 'k'] and domain is not domain_o_xy:
                self.fft_xy(**kwargs)

    def fft_z(self, method='mp', nthread=multiprocessing.cpu_count(),
              **kwargs):  # move to another domain ( time<->frequency )
        _logger.debug('calculating dfl fft_z from ' + self.domain_z + ' domain with ' + method)
        start = time.time()
        orig_domain = self.domain_z
        
        if nthread < 2:
            method = 'np'
        
        if orig_domain == 't':
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.fft(self.fld, axis=0, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                               threads=nthread, auto_align_input=False, auto_contiguous=False,
                                               avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.fft(self.fld, axis=0)
            # else:
            #     raise ValueError('fft method should be "np" or "mp"')
            self.fld = np.fft.ifftshift(self.fld, 0)
            self.fld /= np.sqrt(self.Nz())
            self.domain_z = 'f'
        elif orig_domain == 'f':
            self.fld = np.fft.fftshift(self.fld, 0)
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.ifft(self.fld, axis=0, overwrite_input=True, planner_effort='FFTW_ESTIMATE',
                                                threads=nthread, auto_align_input=False, auto_contiguous=False,
                                                avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.ifft(self.fld, axis=0)
                
                # else:
                # raise ValueError("fft method should be 'np' or 'mp'")
            self.fld *= np.sqrt(self.Nz())
            self.domain_z = 't'
        else:
            raise ValueError("domain_z value should be 't' or 'f'")
        
        t_func = time.time() - start
        if t_func < 60:
            _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
        else:
            _logger.debug(ind_str + 'done in %.2f min' % (t_func / 60))

    def fft_xy(self, method='mp', nthread=multiprocessing.cpu_count(),
               **kwargs):  # move to another domain ( spce<->inverse_space )
        _logger.debug('calculating fft_xy from ' + self.domain_xy + ' domain with ' + method)
        start = time.time()
        domain_orig = self.domain_xy

        if nthread < 2:
            method = 'np'
        
        if domain_orig == 's':
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.fft2(self.fld, axes=(1, 2), overwrite_input=False,
                                                planner_effort='FFTW_ESTIMATE', threads=nthread, auto_align_input=False,
                                                auto_contiguous=False, avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.fft2(self.fld, axes=(1, 2))
                # else:
                # raise ValueError("fft method should be 'np' or 'mp'")
            self.fld = np.fft.fftshift(self.fld, axes=(1, 2))
            self.fld /= np.sqrt(self.Nx() * self.Ny())
            self.domain_xy = 'k'
        elif domain_orig == 'k':
            self.fld = np.fft.ifftshift(self.fld, axes=(1, 2))
            if method == 'mp' and fftw_avail:
                fft_exec = pyfftw.builders.ifft2(self.fld, axes=(1, 2), overwrite_input=False,
                                                 planner_effort='FFTW_ESTIMATE', threads=nthread,
                                                 auto_align_input=False, auto_contiguous=False, avoid_copy=True)
                self.fld = fft_exec()
            else:
                self.fld = np.fft.ifft2(self.fld, axes=(1, 2))
            # else:
            #     raise ValueError("fft method should be 'np' or 'mp'")
            self.fld *= np.sqrt(self.Nx() * self.Ny())
            self.domain_xy = 's'
        
        else:
            raise ValueError("domain_xy value should be 's' or 'k'")
        
        t_func = time.time() - start
        if t_func < 60:
            _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
        else:
            _logger.debug(ind_str + 'done in %.2f min' % (t_func / 60))
    

    def mut_coh_func(self, norm=1, jit=1):
        '''
        calculates mutual coherence function
        consider downsampling the field first
        '''
        if jit:
            J = np.zeros([self.Ny(), self.Nx(), self.Ny(), self.Nx()]).astype(np.complex128)
            mut_coh_func(J, self.fld, norm=norm)
        else:
            I = self.int_xy() / self.Nz()
            J = np.mean(
                self.fld[:, :, :, np.newaxis, np.newaxis].conjugate() * self.fld[:, np.newaxis, np.newaxis, :, :],
                axis=0)
            if norm:
                J /= (I[:, :, np.newaxis, np.newaxis] * I[np.newaxis, np.newaxis, :, :])
        return J
    
    def coh(self, jit=0):
        '''
        calculates degree of transverse coherence
        consider downsampling the field first
        '''
        I = self.int_xy() / self.Nz()
        J = self.mut_coh_func(norm=0, jit=jit)
        coh = np.sum(abs(J) ** 2) / np.sum(I) ** 2
        return coh
        
    def tilt(self, angle=0, plane='x', return_orig_domains=True):
        '''
        deflects the radaition in given direction by given angle
        by introducing transverse phase chirp
        '''
        _logger.info('tilting radiation by {:.4e} rad in {} plane'.format(angle, plane))
        _logger.warn(ind_str + 'in beta')
        angle_warn = ind_str + 'deflection angle exceeds inverse space mesh range'
        
        k = 2 * pi / self.xlamds
        domains = self.domains()
        
        self.to_domain('s')
        if plane == 'y':
            if np.abs(angle) > self.xlamds / self.dy / 2:
                _logger.warning(angle_warn)
            dphi =  angle * k * self.scale_y()
            self.fld = self.fld * np.exp(1j * dphi)[np.newaxis, :, np.newaxis]
        elif plane == 'x':
            if np.abs(angle) > self.xlamds / self.dx / 2:
                _logger.warning(angle_warn)
            dphi =  angle * k * self.scale_x()
            self.fld = self.fld * np.exp(1j * dphi)[np.newaxis, np.newaxis, :]
        else:
            raise ValueError('plane should be "x" or "y"')
            
        if return_orig_domains:
            self.to_domain(domains)
    
            
    def disperse(self, disp=0, E_ph0=None, plane='x', return_orig_domains=True):
        '''
        introducing angular dispersion in given plane by deflecting the radaition by given angle depending on its frequency
        disp is the dispertion coefficient [rad/eV]
        E_ph0 is the photon energy in [eV] direction of which would not be changed (principal ray)
        '''
        _logger.info('introducing dispersion of {:.4e} [rad/eV] in {} plane'.format(disp, plane))
        _logger.warn(ind_str + 'in beta')
        angle_warn = ind_str + 'deflection angle exceeds inverse space mesh range'
        if E_ph0 == None:
            E_ph0 = 2 *np.pi / self.xlamds * speed_of_light * hr_eV_s
        
        dk = 2 * pi / self.Lz()
        k = 2 * pi / self.xlamds        
        phen = np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz()) * speed_of_light * hr_eV_s
        angle = disp * (phen - E_ph0)
        
        if np.amax([np.abs(np.min(angle)), np.abs(np.max(angle))]) > self.xlamds / self.dy / 2:
            _logger.warning(angle_warn)
        
        domains = self.domains()
        self.to_domain('sf')
        if plane =='y':
            dphi =  angle[:,np.newaxis] * k * self.scale_y()[np.newaxis, :]
            self.fld = self.fld * np.exp(1j *dphi)[:, :, np.newaxis]
        elif plane == 'x':
            dphi =  angle[:,np.newaxis] * k * self.scale_x()[np.newaxis, :]
            self.fld = self.fld * np.exp(1j *dphi)[:, np.newaxis, :]
        
        if return_orig_domains:
            self.to_domain(domains)
            
    def curve_wavefront(self, r=np.inf, plane='xy', domain_z=None):
        """
        introduction of the additional
        wavefront curvature with radius r

        r can be scalar or vector with self.Nz() points
        r>0 -> converging wavefront

        plane is the plane in which wavefront is curved:
            'x' - horizontal focusing
            'y' - vertical focusing
            'xy' - focusing in both planes

        domain_z is the domain in which wavefront curvature is introduced
            'f' - frequency
            't' - time
            None - original domain (default)

        """

        domains = domain_o_z, domain_o_xy = self.domain_z, self.domain_xy

        if domain_z == None:
            domain_z = domain_o_z

        _logger.debug('curving radiation wavefront by {}m in {} domain'.format(r, domain_z))

        if np.size(r) == 1:
            if r == 0:
                _logger.error(ind_str + 'radius of curvature should not be zero')
                raise ValueError('radius of curvature should not be zero')
            elif r == np.inf:
                _logger.debug(ind_str + 'radius of curvature is infinite, skipping')
                return
            else:
                pass

        if domain_z == 'f':
            self.to_domain('fs')
            x, y = np.meshgrid(self.scale_x(), self.scale_y())
            if plane == 'xy' or plane == 'yx':
                arg2 = x ** 2 + y ** 2
            elif plane == 'x':
                arg2 = x ** 2
            elif plane == 'y':
                arg2 = y ** 2
            else:
                _logger.error('"plane" should be in ["x", "y", "xy"]')
                raise ValueError()
            k = 2 * np.pi / self.scale_z()
            if np.size(r) == 1:
                self.fld *= np.exp(-1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / r)
            elif np.size(r) == self.Nz():
                self.fld *= np.exp(
                    -1j * k[:, np.newaxis, np.newaxis] / 2 * arg2[np.newaxis, :, :] / r[:, np.newaxis, np.newaxis])

        elif domain_z == 't':
            self.to_domain('ts')
            x, y = np.meshgrid(self.scale_x(), self.scale_y())
            if plane == 'xy' or plane == 'yx':
                arg2 = x ** 2 + y ** 2
            elif plane == 'x':
                arg2 = x ** 2
            elif plane == 'y':
                arg2 = y ** 2
            else:
                _logger.error('"plane" should be in ["x", "y", "xy"]')
                raise ValueError()
            k = 2 * np.pi / self.xlamds
            if np.size(r) == 1:
                self.fld *= np.exp(-1j * k / 2 * arg2 / r)[np.newaxis, :, :]
            elif np.size(r) == self.Nz():
                self.fld *= np.exp(-1j * k / 2 * arg2[np.newaxis, :, :] / r[:, np.newaxis, np.newaxis])
            else:
                raise ValueError('wrong dimensions of radius of curvature')
        else:
            ValueError('domain_z should be in ["f", "t", None]')

        self.to_domain(domains)

    def prop(self, z, fine=1, return_result=0, return_orig_domains=1, **kwargs):
        """
        Angular-spectrum propagation for fieldfile
        
        can handle wide spectrum
          (every slice in freq.domain is propagated
           according to its frequency)
        no kx**2+ky**2<<k0**2 limitation
        
        dfl is the RadiationField() object
        z is the propagation distance in [m]
        fine=1 is a flag for ~2x faster propagation.
            no Fourier transform to frequency domain is done
            assumes no angular dispersion (true for plain FEL radiation)
            assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
        
        return_result does not modify self, but returns result
        
        z>0 -> forward direction
        """
        _logger.info('propagating dfl file by %.2f meters' % (z))
        
        if z == 0:
            _logger.debug(ind_str + 'z=0, returning original')
            if return_result:
                return self
            else:
                return
        
        start = time.time()
        
        domains = self.domains()
        
        if return_result:
            copydfl = deepcopy(self)
            copydfl, self = self, copydfl
        
        if fine == 1:
            self.to_domain('kf')
        elif fine == -1:
            self.to_domain('kt')
        else:
            self.to_domain('k')
        
        if self.domain_z == 'f':
            k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
            k = self.scale_kz()
            # H = np.exp(1j * z * (np.sqrt((k**2)[:,np.newaxis,np.newaxis] - (k_x**2)[np.newaxis,:,:] - (k_y**2)[np.newaxis,:,:]) - k[:,np.newaxis,np.newaxis]))
            # self.fld *= H
            for i in range(self.Nz()):  # more memory efficient
                H = np.exp(1j * z * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i]))
                self.fld[i, :, :] *= H
        else:
            k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
            k = 2 * np.pi / self.xlamds
            H = np.exp(1j * z * (np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2) - k))
            # self.fld *= H[np.newaxis,:,:]
            for i in range(self.Nz()):  # more memory efficient
                self.fld[i, :, :] *= H
        
        if return_orig_domains:
            self.to_domain(domains)
        
        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        if return_result:
            copydfl, self = self, copydfl
            return copydfl
        
    def prop_m(self, z, m=1, fine=1, return_result=0, return_orig_domains=1, **kwargs):
        """
        Angular-spectrum propagation for fieldfile
        
        can handle wide spectrum
          (every slice in freq.domain is propagated
           according to its frequency)
        no kx**2+ky**2<<k0**2 limitation
        
        dfl is the RadiationField() object
        z is the propagation distance in [m]
        m is the output mesh size in terms of input mesh size (m = L_out/L_inp)
        which can be a number m or a pair of number m = [m_x, m_y]
        fine==0 is a flag for ~2x faster propagation.
            no Fourier transform to frequency domain is done
            assumes no angular dispersion (true for plain FEL radiation)
            assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
        
        z>0 -> forward direction
        """
        _logger.info('propagating dfl file by %.2f meters' % (z))
        
        start = time.time()
        domains = self.domains()
        
        if return_result:
            copydfl = deepcopy(self)
            copydfl, self = self, copydfl
        
        domain_z = self.domain_z
        if np.size(m)==1:
            m_x = m
            m_y = m
        elif np.size(m)==2:
            m_x = m[0]
            m_y = m[1]
        else:
            _logger.error(ind_str + 'm mast have shape = 1 or 2')
            raise ValueError('m mast have shape = 1 or 2')
             
        if z==0:
            _logger.debug(ind_str + 'z=0, returning original')
            if m_x != 1 and m_y != 1:
                _logger.debug(ind_str + 'mesh is not resized in the case z = 0')
            if return_result:
                return self
            else:
                return
        
        if m_x != 1:
            self.curve_wavefront(-z / (1 - m_x), plane='x')
        if m_y != 1:
            self.curve_wavefront(-z / (1 - m_y), plane='y')
        
        if fine == 1:
            self.to_domain('kf')
        elif fine == -1:
            self.to_domain('kt')
        else:
            self.to_domain('k')
        
        if z != 0:
            H = 1
            if self.domain_z == 'f':
                k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
                k = self.scale_kz()
                # H = np.exp(1j * z * (np.sqrt((k**2)[:,np.newaxis,np.newaxis] - (k_x**2)[np.newaxis,:,:] - (k_y**2)[np.newaxis,:,:]) - k[:,np.newaxis,np.newaxis]))
                # self.fld *= H
                #for i in range(self.Nz()):
                #    H = np.exp(1j * z / m * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i]))
                #    self.fld[i, :, :] *= H
                if m_x != 0:
                    for i in range(self.Nz()):
                        H=np.exp(1j * z / m_x * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i]))
                        self.fld[i, :, :] *= H
                if m_y != 0:
                    for i in range(self.Nz()):
                        H=np.exp(1j * z / m_y * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i]))
                        self.fld[i, :, :] *= H           
            else:
                k_x, k_y = np.meshgrid(self.scale_kx(), self.scale_ky())
                k = 2 * np.pi / self.xlamds
                if m_x != 0:
                    H*=np.exp(1j * z / m_x * (np.sqrt(k ** 2 - k_x ** 2) - k))                
                if m_y != 0:
                    H*=np.exp(1j * z / m_y * (np.sqrt(k ** 2 - k_y ** 2) - k))
                for i in range(self.Nz()):
                    self.fld[i, :, :] *= H
        
        self.dx *= m_x
        self.dy *= m_y
        
        if return_orig_domains:
            self.to_domain(domains)
        if m_x != 1:
            self.curve_wavefront(-m_x * z / (m_x - 1), plane='x')
        if m_y != 1:
            self.curve_wavefront(-m_y * z / (m_y - 1), plane='y')
        
        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % (t_func))
        
        if return_result:
            copydfl, self = self, copydfl
            return copydfl
        
class HeightProfile: # this one is here because generate_1d_profile is a method
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
    
def dfldomain_check(domains, both_req=False):
    err = ValueError(
        'domains should be a string with one or two letters from ("t" or "f") and ("s" or "k"), not {}'.format(
            str(domains)))

    # if type(domains) is not str:
    #     raise err
    if len(domains) < 1 or len(domains) > 2:
        raise err
    if len(domains) < 2 and both_req == True:
        raise ValueError('please provide both domains, e.g. "ts" "fs" "tk" "fk"')

    domains_avail = ['t', 'f', 's', 'k']
    for letter in domains:
        if letter not in domains_avail:
            raise err

    if len(domains) == 2:
        D = [['t', 'f'], ['s', 'k']]
        for d in D:
            if domains[0] in d and domains[1] in d:
                raise err

        """
        tranfers radiation to specified domains
        *domains is a string with one or two letters: 
            ("t" or "f") and ("s" or "k")
        where 
            't' (time); 'f' (frequency); 's' (space); 'k' (inverse space); 
        e.g.
            't'; 'f'; 's'; 'k'; 'ts'; 'fs'; 'tk'; 'fk'
        order does not matter
        
        **kwargs are passed down to self.fft_z and self.fft_xy
        """                

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
    """
    Matchs OpticalElement object to its mask. Several masks can correspond to one OpticalElement object 
    The function rewrites OpticalElement object parameters to Mask object parameters
    
    :param element: OpticsElement class
    """
    element.mask = Mask()

    if element.__class__ is None:
        raise ValueError('Optics element must belong to the OpticsElement class')

    elif element.__class__ is FreeSpace: #implementation of several masks
        if element.method in ['PropMask', 'PropMask_kf', 'PropMask_kt']:
            if element.mx == 1 and element.my == 1:
                element.mask = PropMask(element.l, element.method)                
            elif element.mx != 1 and element.my != 1:
                element.mask = Prop_mMask(element.l, element.mx, element.my, element.method)
            else:
                ValueError("mx and my must be positive non-zero values")
        elif element.type == 'Fraunhofer_Propagator':
            _logger.warn('Fraunhofer Propagation method has not implemented yet')
            pass    
        elif element.type == 'Fresnel_Propagator':
            _logger.warn('Fresnel Propagation method has not implemented yet')          
            pass                               
        else:            
            raise ValueError('Propagator method can be PropMask (see Tutorials), Fraunhofer_Propagator or Fresnel_Propagator')    
        
    elif element.__class__ is ThinLens: 
        element.mask = LensMask(element.fx, element.fy)
  
    elif element.__class__ is ApertureRect: #no cheked
        mask = ApertureRectMask()
        mask.lx = element.lx
        mask.ly = element.ly
        mask.cx = element.cx
        mask.cy = element.cy
        element.mask = mask
        
    elif element.__class__ is ApertureEllips: #no checked
        mask = ApertureEllipsMask() #which way is better?
        mask.ax = element.ax
        mask.ay = element.ay
        mask.cx = element.cx
        mask.cy = element.cy
        element.mask = mask
      
    elif element.__class__ is DispersiveSection: #no checked
        element.mask = PhaseDelayMask(element.coeff, element.E_ph0)
        
    elif element.__class__ is ImperfectMirror: #no checked
        element.mask = MirrorMask(element.height_profile, element.hrms, element.angle, element.plane, element.lx, element.ly)

    else:
        raise ValueError('Optics element must belong to one of the child OpticsElement classes')
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
    """
    Parent class Optics element
    :param eid: element id, (for example 'KB')  
    """
    def __init__(self, eid=None):
        self.eid = eid
        
    def apply(self, dfl): #is this method need?
        """
        TODO
        write documentation
        """
        get_transfer_function(self)
        self.mask.apply(self.mask, dfl)
           
class FreeSpace(OpticsElement):
    """
    Class for Drift element 
    :param OpticsElement(): optics element parent class with following parameters
        :param eid: element id, (for example 'KB')  
    :param l: propagation distance
    :param mx: is the output x mesh size in terms of input mesh size (mx = Lx_out/Lx_inp)
    :param my: is the output y mesh size in terms of input mesh size (my = Ly_out/Ly_inp)
    """
    def __init__(self, l=0., mx=1, my=1, method='PropMask_kf', eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.l = l
        self.mx = mx
        self.my = my
        self.method = method  #method of propagation, also may be 
                                # 'Fraunhofer_Propagator'
                                # 'Fresnel_Propagator' . . .

class ThinLens(OpticsElement):
    """
    Class for Lens element
    :param OpticsElement(): optics element parent class with following parameters
        :param eid: element id, (for example 'KB') 
    :param fx: focus length in x direction
    :param fy: focus length in y direction
    """
    def __init__(self, fx=np.inf, fy=np.inf, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.fx = fx
        self.fy = fy
        
class Aperture(OpticsElement):
    """
    Aperture
    write documentation
    """
    def __init__(self, eid=None):
        OpticsElement.__init__(self, eid=eid)
        
class ApertureRect(Aperture):
    """
    Rectangular aperture
    
    write documentation
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
    ax =: ellipse x main axis
    ay =: ellipse y main axis
    cx =: ellipse x coordinate of center 
    cy =: ellipse x coordinate of center 
    eid =: - id of the optical element
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
        coeff --- 
        coeff[0] =: measured in [rad]      --- phase
        coeff[1] =: measured in [fm s ^ 1] --- group delay
        coeff[2] =: measured in [fm s ^ 2] --- group delay dispersion (GDD)
        coeff[3] =: measured in [fm s ^ 3] --- third-order dispersion (TOD)
        ...
        E_ph0 --- energy with respect to which the phase shift is calculated
        eid --- id of the optical element
    """  
    def __init__(self, coeff=[0], E_ph0=None, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.coeff = coeff
        self.E_ph0 = E_ph0


class ImperfectMirror(OpticsElement):
    """
    TODO
    write documentation
    """
    def __init__(self, height_profile=None, hrms=0, lx=np.inf, ly=np.inf, angle=np.pi * 2 / 180, plane='x', eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.height_profile = height_profile
        self.hrms = hrms
        self.lx = lx
        self.ly = ly
        self.angle=angle
        self.plane=plane

### goes to transfer_function.py ### may be arrized and replaced by a line "from ocelot.rad.transfer_function import * "
        
class Mask(Grid):
    """
    Mask element class
    The class represents a transfer function of an optical element. Note, several masks can correspond to one optical element.
    :param Grid: with following parameters  
        :params (dx, dy, dz): spatial step of the mesh 
        :param shape: number of point in each direction of 3D data mesh
        :param xlamds: carrier wavelength of the wavepacket
        :param used_aprox: the approximation used in solving the wave equation. OCELOT works in Slowly Varying Amplitude Approximation
    :param domain_z: longitudinal domain (t - time, f - frequency)   
    :param domain_x: transverse domain (s - space, k - inverse space)      
    :param domain_y: transverse domain (s - space, k - inverse space)       
    """
    def __init__(self, shape=(0, 0, 0)):
        Grid.__init__(self, shape=shape)
        self.domain_z = 't'   # longitudinal domain (t - time, f - frequency)
        self.domain_x = 's'   # transverse domain (s - space, k - inverse space)
        self.domain_y = 's'   # transverse domain (s - space, k - inverse space)
        
    def __mul__(self, other):#check this stuff when it will been needed
        """
        Multiplication operator for two masks 
        """
        m = deepcopy(self)
        if other.__class__ in [self] and self.mask is not None and other.mask is not None:
            m.mask = self.mask * other.mask
            return m
        else: ValueError("'other' must belong to Mask class") 

class PropMask(Mask):
    """
    Angular-spectrum propagation for fieldfile
    
    can handle wide spectrum
      (every slice in freq.domain is propagated
       according to its frequency)
    no kx**2+ky**2<<k0**2 limitation
    
    dfl is the RadiationField() object
    
    z0 is the propagation distance in [m]
    for 'kt' domain propagation
        no Fourier transform to frequency domain is done
        assumes no angular dispersion (true for plain FEL radiation)
        assumes narrow spectrum at center of xlamds (true for plain FEL radiation)
    'kf' propagation is a default option
    
    z>0 -> forward direction
    z<0 -> backward direction
    z=0 return original
    """   
    def __init__(self, z0, method='PropMask_kf'):
        Mask.__init__(self)
        self.z0 = z0 
        self.method = method
        self.mask = None

    def apply(self, dfl):
        """
        'apply' method for PropMask class 
        
        transform dfl to the domain of the propagation 'kf' or 'kt'
        get the transfer function calling 'get_mask' method, e.g. get H transfer function 
        multiply dfl and the mask, e.g. E = E_0 * H convolution in inverse space domain
        return the field to the original domain
        """
        
        _logger.info('propagating dfl file by %.2f meters' % (self.z0))
        
        if self.z0 == 0:
            _logger.debug(ind_str + 'z0=0, returning original')
            return dfl

        start = time.time()

        domains = dfl.domains()
        self.domain_x = 'k'
        self.domain_y = 'k'
        if self.method in ['PropMask', 'PropMask_kf']:
            dfl.to_domain('kf')
            self.domain_x = 'f'
        elif self.method == 'PropMask_kt':
            dfl.to_domain('kt')
            self.domain_x = 't'
        else:
            raise ValueError("propagation method should be 'PropMask', 'PropMask_kf' or 'PropMask_kt'")
                         
        if self.mask is None:
            self.get_mask(dfl) # get H transfer function 

        dfl.fld *= self.mask # E = E_0 * H convolution in inverse space domain
        
        dfl.to_domain(domains) # back to original domain  

        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        return dfl
        
    def get_mask(self, dfl):
        """
        'get_mask' method for PropMask class
        find the transfer function for propagation in 'kf' of 'ks' domains
        H = exp(iz0(k^2 - kx^2 - ky^2)^(1/2) - k) 
        """
        self.copy_grid(dfl)
       
        k_x, k_y = np.meshgrid(self.grid_kx(), self.grid_ky())
        
        if dfl.domain_z == 'f':
            k = self.grid_kz()
            self.mask = [np.exp(1j * self.z0 * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())] 
        elif dfl.domain_z == 't':
            k = 2 * np.pi / self.xlamds
            self.mask = [np.exp(1j * self.z0 * (np.sqrt(k ** 2 - k_x ** 2 - k_y ** 2) - k)) for i in range(dfl.Nz())]
        else: 
            raise ValueError('wrong field domain, domain must be ks or kf ')    
            
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
    def __init__(self, z0, mx, my, method):
        Mask.__init__(self)
        self.z0 = z0
        self.mx = mx
        self.my = my
        self.method = method
        self.mask = None

    def apply(self, dfl):
        """
        'apply' method for Prop_mMask class 
        
        transform dfl to the domain of the propagation 'kf' or 'kt'
        transforming the wavefront curvature transform by a phase factor z0/(1 - m)
        get the transfer function calling 'get_mask' method, e.g. get H transfer function 
        multiply dfl and the mask, e.g. E = E_0 * H convolution in inverse space domain
        transverce mesh step resizing 
        return the field to the original domain
        transforming the wavefront curvature transform by a phase factor -m*z0/(m - 1) --- inverse transformation
        """   
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
        self.domain_x = 'k'
        self.domain_y = 'k'
        if self.method in ['PropMask', 'PropMask_kf']:
            dfl.to_domain('kf')
            self.domain_x = 'f'
        elif self.method == 'PropMask_kt':
            dfl.to_domain('kt')
            self.domain_x = 't'
        else:
            raise ValueError("propagation method should be 'PropMask', 'PropMask_kf' or 'PropMask_kt'")
        
        if self.mask is None:
            self.get_mask(dfl) # get H transfer function 
            
        dfl.fld *= self.mask #E = E_0 * H convolution in inverse space domain
               
        dfl.dx *= self.mx #transform grid to output mesh size
        dfl.dy *= self.my        
        self.dx *= self.mx #transform grid to output mesh size
        self.dy *= self.my   
        
        print(id(dfl.dx), id(self.dx))
        
        dfl.to_domain(domains) # back to the original domain
        
        if self.mx != 1:
#            dfl.curve_wavefront(-self.mx * self.z0 / (self.mx - 1), plane='x')
            dfl = QuadCurvMask(r = -self.mx * self.z0 / (self.mx - 1), plane='x').apply(dfl)
            print('aaaa')
        if self.my != 1:
#            dfl.curve_wavefront(-self.my * self.z0 / (self.my - 1), plane='y')
            dfl = QuadCurvMask(r = -self.my * self.z0 / (self.my - 1), plane='y').apply(dfl)

        t_func = time.time() - start
        _logger.debug(ind_str + 'done in %.2f sec' % t_func)
        
        return dfl
    
    def get_mask(self, dfl):
        """
        'get_mask' method for Prop_mMask class
        find the transfer function for propagation in 'kf' of 'ks' domains
        H = exp(iz0(k^2 - kx^2 - ky^2)^(1/2) - k) 
        """
        self.copy_grid(dfl)
        
        k_x, k_y = np.meshgrid(self.grid_kx(), self.grid_ky())
        if dfl.domain_z == 'f':
            self.mask = np.ones(dfl.shape, dtype=complex128)
            k = self.grid_kz()
            if self.mx != 0:
#                self.mask[i,:,:] *= [np.exp(1j * self.z0/self.mx * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i])) for i in range(dfl.Nz())] #Hx = exp(iz0/mx(k^2 - kx^2)^(1/2) - k)
                for i in range(self.Nz()):
                    self.mask[i,:,:] *= np.exp(1j * self.z0 / self.mx * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i]))
            if self.my != 0:
#                self.mask[i,:,:] *= [np.exp(1j * self.z0/self.my * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())] #Hy = exp(iz0/my(k^2 - ky^2)^(1/2) - k)                   
                for i in range(self.Nz()):
                    self.mask[i,:,:] *= np.exp(1j * self.z0 / self.my * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i]))
        elif dfl.domain_z == 't':
            k = 2 * np.pi / self.xlamds
            Hx = [np.exp(1j * self.z0/self.mx * (np.sqrt(k ** 2 - k_x ** 2) - k)) for i in range(dfl.Nz())] 
            Hy = [np.exp(1j * self.z0/self.my * (np.sqrt(k ** 2 - k_y ** 2) - k)) for i in range(dfl.Nz())]          
            self.mask = Hx*Hy
        else: 
            raise ValueError("wrong field domain, domain must be 'ks' or 'kf'")    
            
        return self.mask    
    
class QuadCurvMask(Mask):
    """
    Quadratic curvature wavefront mask
    add quadratic x and y term in 's' space 
    
    :param r: radius of curveture
    :param plane: plane in which the curvature added
    :param mask: the transfer function itself
    """
    def __init__(self, r, plane):
        Mask.__init__(self)
        self.r = r #is the radius of curvature
        self.plane = plane #is the plane in which the curvature is applied
        self.mask = None #is the transfer function itself

    def apply(self, dfl):
        """
        TODO 
        add documentation
        """
        domains = dfl.domain_z, dfl.domain_xy
        
        self.domain_x = 's'
        self.domain_y = 's'
        if self.domain_z is None:
            self.domain_z = dfl.domain_z
        
        _logger.debug('curving radiation wavefront by {}m in {} domain'.format(self.r, self.domain_z))
        
        if np.size(self.r) == 1:
            if self.r == 0:
                raise ValueError('radius of curvature cannot be zero')
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
        """
        TODO 
        add documentation
        """        
        self.copy_grid(dfl)
       
        x, y = np.meshgrid(self.grid_x(), self.grid_y())
        if self.plane == 'xy' or self.plane == 'yx':
            arg2 = x ** 2 + y ** 2
        elif self.plane == 'x':
            arg2 = x ** 2
        elif self.plane == 'y':
            arg2 = y ** 2
        else:
            raise ValueError("'plane' should be in 'x', 'y'")

        if self.domain_z == 'f':
            k = 2 * np.pi /dfl.grid_z() 

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
            raise ValueError("domain_z should be in 'f', 't'")
            
        return self.mask

class ApertureRectMask(Mask):
    """
    TODO
    write documentation
    add logging
    """
    def __init__(self, lx=np.inf, ly=np.inf, cx=0, cy=0, shape=(0, 0, 0)):
        Mask.__init__(self, shape=shape)
        self.lx = lx
        self.ly = ly
        self.cx = cx
        self.cy = cy
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
        
        self.copy_grid(dfl)
            
        if np.size(self.lx) == 1:
            self.lx = [-self.lx / 2, self.lx / 2]
        if np.size(self.ly) == 1:
            self.ly = [-self.ly / 2, self.ly / 2]
        _logger.debug(ind_str + 'ap_x = {}'.format(self.lx))
        _logger.debug(ind_str + 'ap_y = {}'.format(self.ly ))

        idx_x = np.where((self.grid_x() >= self.lx[0]) & (self.grid_x() <= self.lx[1]))[0]
        idx_x1 = idx_x[0]
        idx_x2 = idx_x[-1]

        idx_y = np.where((self.grid_y() >= self.ly [0]) & (self.grid_y() <= self.ly [1]))[0]
        idx_y1 = idx_y[0]
        idx_y2 = idx_y[-1]

        _logger.debug(ind_str + 'idx_x = {}-{}'.format(idx_x1, idx_x2))
        _logger.debug(ind_str + 'idx_y = {}-{}'.format(idx_y1, idx_y2))

        self.mask = np.zeros_like(dfl.fld[0, :, :])
        self.mask[idx_y1:idx_y2, idx_x1:idx_x2] = 1
        return self.mask

class ApertureEllipsMask(Mask):
    """
    TODO
    write documentation
    add logging
    """
    def __init__(self, shape=(0,0,0)):
        Mask.__init__(self, shape=shape)
        self.ax = np.inf
        self.ay = np.inf
        self.cx = 0
        self.cy = 0
        self.mask = None 
    
    def ellipse(self, dfl):    
        x, y = np.meshgrid(self.grid_x(), self.grid_y())
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
        
        self.copy_grid(dfl)
        
        self.mask = np.zeros_like(dfl.fld[0, :, :])
        self.mask[self.ellipse(dfl) <= 1] = 1
        return self.mask
    
class LensMask(Mask):
    """
    TODO
    write documentation
    add logging
    """
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

        _logger.info('get the lens mask')        
        H_fx = QuadCurvMask(r=self.fx, plane='x').get_mask(dfl)        
        H_fy = QuadCurvMask(r=self.fy, plane='y').get_mask(dfl)
        self.mask = H_fx * H_fy
      
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

        _logger.info('get the frequency chirp mask')
        
        self.copy_grid(dfl)
        
        if self.E_ph0 == None:
            w0 = 2 * np.pi * speed_of_light / self.xlamds
    
        elif self.E_ph0 == 'center':
            _, lamds = np.mean([dfl.int_z(), self.gird_z()], axis=(1))
            w0 = 2 * np.pi * speed_of_light / lamds
    
        elif isinstance(self.E_ph0, str) is not True:
            w0 = 2 * np.pi * self.E_ph0 / h_eV_s
    
        else:
            raise ValueError("E_ph0 must be None or 'center' or some value")

        w = self.grid_kz() * speed_of_light
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
    
    TODO
    write documentation
    add logging
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
            RectApp = ApertureRectMask()
            RectApp.lx = self.lx
            RectApp.ly = self.ly           
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
        
        self.copy_grid(dfl)
        
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
    
#%%    
### script itself ###

#optics elements check
dfl = RadiationField()
E_pohoton = 8000 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'shape':(101,101,1),           #(x,y,z) shape of field matrix (reversed) to dfl.fld
        'dgrid':(500e-6,500e-6,1e-6), #(x,y,z) [m] - size of field matrix
        'power_rms':(10e-6,10e-6,0.1e-6),#(x,y,z) [m] - rms size of the radiation distribution (gaussian)
        'power_center':(0,0,None),     #(x,y,z) [m] - position of the radiation distribution
        'power_angle':(0,0),           #(x,y) [rad] - angle of further radiation propagation
        'power_waistpos':(0,0),     #(Z_x,Z_y) [m] downstrean location of the waist of the beam
        'wavelength':None,             #central frequency of the radiation, if different from xlamds
        'zsep':None,                   #distance between slices in z as zsep*xlamds
        'freq_chirp':0,                #dw/dt=[1/fs**2] - requency chirp of the beam around power_center[2]
        'en_pulse':None,               #total energy or max power of the pulse, use only one
        'power':1e6,
        }
dfl = generate_gaussian_dfl(**kwargs);
dfl1 = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition
dfl2 = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition

#%%
###lens checking
f=100
lens = ThinLens(fx=f, fy=f)
line1 = (lens)
lat1 = OpticsLine(line1)
dfl = propagate(lat1, dfl)

dfl1.curve_wavefront(r=f)

plot_dfl(dfl, fig_name='after3', phase=1)
plot_dfl(dfl1, fig_name='after4', phase=1)
#%%
#PropMask and Prop_mMask cheking
#plot_dfl(dfl1, fig_name='before1', phase=1)
#
#d1 = FreeSpace(l=1000, mx=1, my=1)
d2 = FreeSpace(l=50, mx=3, my=3)
#
#line1 = (d1)
line2 = (d2)
#
#lat1 = OpticsLine(line1)
lat2 = OpticsLine(line2)
#
#dfl1 = propagate(lat1, dfl1)
dfl2 = propagate(lat2, dfl2)
#
#plot_dfl(dfl1, fig_name='after1', phase=1)
plot_dfl(dfl2, fig_name='after2', phase=1)

#dfl1 = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition
dfl3 = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition

#dfl3.prop(z=50)
dfl3.prop_m(z=50, m=3)

#plot_dfl(dfl1, fig_name='after3', phase=1)
plot_dfl(dfl3, fig_name='after4', phase=1)
#%%
'''
#dfl = generate_gaussian_dfl(1239.8/500*1e-9, shape=(3,3,501), dgrid=(1e-3,1e-3,400e-6), power_rms=(0.5e-3,0.5e-3,0.5e-6), 
#                        power_center=(0,0,None), power_angle=(0,0), power_waistpos=(0,0), #wavelength=[4.20e-9,4.08e-9], 
#                        zsep=None, freq_chirp=0, energy=None, power=10e6, debug=1)

#wig = wigner_dfl(dfl)
#plot_wigner(wig, fig_name = 'wigner_before', plot_moments=0)

plot_dfl(dfl, fig_name='before', phase=1)

appRect =  ApertureRect(lx=0.001, ly=0.001, cx=0, cy=0)
appEl = ApertureEllips(ax=0.001, ay=0.001, cx=0, cy=0)
l = ThinLens(fx=25, fy=25)
d = FreeSpace(l=200, mx=1, my=1)
PhaseDel = DispersiveSection(coeff=[0,0,-100, -100])
Mirror = ImperfectMirror(hrms=1e-9)
#
line = (Mirror)#, app)
lat = OpticsLine(line)

dfl = propagate(lat, dfl)


#wig = wigner_dfl(dfl)
#plot_wigner(wig, fig_name = 'wigner_after', plot_moments=0)

plot_dfl(dfl, fig_name='after1', phase=1)

'''







