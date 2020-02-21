import logging
import time

import numpy as np
from copy import deepcopy
from numpy import inf, complex128, complex64
from math import factorial

import ocelot
from ocelot.common.globals import *
from ocelot import ocelog
from ocelot.common.ocelog import *
_logger = logging.getLogger(__name__) 

from ocelot.optics.wave import dfldomain_check, wigner_dfl, HeightProfile
from ocelot.gui.dfl_plot import plot_dfl, plot_wigner, plot_dfl_waistscan
import multiprocessing
nthread = multiprocessing.cpu_count()

try:
    import pyfftw
    fftw_avail = True
except ImportError:
    print("wave.py: module PYFFTW is not installed. Install it if you want speed up dfl wavefront calculations")
    fftw_avail = False

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
    def __init__(self, shape):
        self.dx = []
        self.dy = []
        self.dz = []
        self.shape = shape
        
        self.xlamds = None
        self.used_aprox = 'SVAE'
        
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

    def grid_t(self):
        return np.linspace(0, self.Lz()/speed_of_light, self.Nz())
    
    def grid_f(self):
        df = 2 * np.pi * speed_of_light / self.Lz() # self.Lz() must be replaced with self.Tz()
        
        if self.xlamds is None:        
            return np.linspace(-df / 2 * self.Nz(), df / 2 * self.Nz(), self.Nz())
        
        elif self.used_aprox == 'SVAE' and self.xlamds is not None:
            f0 = 2 * np.pi * speed_of_light / self.xlamds
            return np.linspace(f0 - df / 2 * self.Nz(), f0 + df / 2 * self.Nz(), self.Nz())
        
        else:
            raise ValueError

    def grid_z(self):
        return self.scale_t() * speed_of_light

    def grid_kz(self):
        return self.grid_f() / speed_of_light
            
class RadiationField(Grid):
    """
    3d or 2d coherent radiation distribution, *.fld variable is the same as Genesis dfl structure
    """

    def __init__(self, shape=(0,0,0)):
        super().__init__(shape=shape)
        self.fld = np.zeros(shape, dtype=complex128)  # (z,y,x)
        self.xlamds = None  # carrier wavelength [m]
        self.domain_z = 't'  # longitudinal domain (t - time, f - frequency)
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

#   old scales for versions compatibility
#   propper scales in meters or 2 pi / meters
    def scale_kx(self):  # scale in meters or meters**-1
        if self.domain_xy == 's':  # space domain
            return np.linspace(-self.Lx() / 2, self.Lx() / 2, self.Nx())
        elif self.domain_xy == 'k':  # inverse space domain
            k = 2 * np.pi / self.dx
            return np.linspace(-k / 2, k / 2, self.Nx())
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_ky(self):  # scale in meters or meters**-1
        if self.domain_xy == 's':  # space domain
            return np.linspace(-self.Ly() / 2, self.Ly() / 2, self.Ny())
        elif self.domain_xy == 'k':  # inverse space domain
            k = 2 * np.pi / self.dy
            return np.linspace(-k / 2, k / 2, self.Ny())
        else:
            raise AttributeError('Wrong domain_xy attribute')

    def scale_kz(self):  # scale in meters or meters**-1
        if self.domain_z == 't':  # time domain
            return np.linspace(0, self.Lz(), self.Nz())
        elif self.domain_z == 'f':  # frequency domain
            dk = 2 * pi / self.Lz()
            k = 2 * pi / self.xlamds
            return np.linspace(k - dk / 2 * self.Nz(), k + dk / 2 * self.Nz(), self.Nz())
        else:
            raise AttributeError('Wrong domain_z attribute')

    def scale_x(self):  # scale in meters or radians
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
        mask = ApertureRectMask()
        mask.lx = element.lx
        mask.ly = element.ly
        mask.cx = element.cx
        mask.cy = element.cy
        element.mask = mask
        
    elif element.__class__ is ApertureEllips:
        mask = ApertureEllipsMask()
        mask.ax = element.ax
        mask.ay = element.ay
        mask.cx = element.cx
        mask.cy = element.cy
        element.mask = mask
      
    elif element.__class__ is ThinLens:
        element.mask = LensMask(element.fx, element.fy)
  
    elif element.__class__ is FreeSpace:
        element.mask = DriftMask(element.l, element.mx, element.my)

    elif element.__class__ is DispersiveSection:
        element.mask = PhaseDelayMask(element.coeff, element.E_ph0)
        
    elif element.__class__ is ImperfectMirror:
        element.mask = MirrorMask(element.height_profile, element.hrms, element.angle, element.plane, element.lx, element.ly)

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
    write documentation
    """
    def __init__(self, eid=None):
        self.eid = eid
        self.domain = "sf"

    def apply(self, dfl): #debag
        get_transfer_function(self)
        self.mask.apply(self.mask, dfl)
           
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
        
class ThinLens(OpticsElement):
    """
    Lens element
    """
    def __init__(self, fx=np.inf, fy=np.inf, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.fx = fx
        self.fy = fy
        
class FreeSpace(OpticsElement):
    """
    Drift element
    write documentation
    """
    def __init__(self, l=0., mx=1, my=1, eid=None):
        OpticsElement.__init__(self, eid=eid)
        self.l = l
        self.mx = mx
        self.my = my
       
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
    
    def __init__(self, shape=(0, 0, 0)):
        Grid.__init__(self, shape=shape)

    def __mul__(self, other):
        m = deepcopy(self)
        if other.__class__ in [self] and self.mask is not None and other.mask is not None:
            m.mask = self.mask * other.mask
            return m
        
    def copy_grid(self, other, version=2):
        if version == 1:
            self.dx = other.dx
            self.dy = other.dy
            self.dz = other.dz
            self.shape = other.shape
            
            self.xlamds = other.xlamds
            self.used_aprox = other.used_aprox
            
        elif version == 2: #copy the same attributes of Mask and RadiationField objects
            attr_list = np.intersect1d(dir(self),dir(other))
            for attr in attr_list:
                if attr.startswith('__') or callable(getattr(self, attr)):
                    continue
                setattr(self, attr, getattr(other, attr))
        else:
            raise ValueError

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

class QuadCurvMask(Mask):
    """
    TODO
    write documentation
    add logging
    """
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
        self.copy_grid(dfl)
       
        x, y = np.meshgrid(self.grid_x(), self.grid_y())
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
     
        self.copy_grid(dfl)
       
        k_x, k_y = np.meshgrid(self.grid_kx(), self.grid_ky())
        
        if dfl.domain_z is 'f':
            k = self.grid_kz()
            self.mask = [np.exp(1j * self.z0 * (np.sqrt(k[i] ** 2 - k_x ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())] #H = exp(iz0(k^2 - kx^2 - ky^2)^(1/2) - k)
        else:
            k = 2 * np.pi / self.xlamds
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

        if dfl.domains == 'kf' or dfl.domains == 'kf' or dfl.domains == 'k':
            dfl.to_domain(dfl.domains) #the field is transformed in inverce space domain and, optionaly, in 'f' or 't' domains
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
        
        self.copy_grid(dfl)
        
        k_x, k_y = np.meshgrid(self.grid_kx(), self.grid_ky())
        k = self.grid_kz()

        if dfl.domains == 'kf':
            k = self.grid_kz()
            Hx = [np.exp(1j * self.z0/self.mx * (np.sqrt(k[i] ** 2 - k_x ** 2) - k[i])) for i in range(dfl.Nz())][0] #Hx = exp(iz0/mx(k^2 - kx^2)^(1/2) - k)
            Hy = [np.exp(1j * self.z0/self.my * (np.sqrt(k[i] ** 2 - k_y ** 2) - k[i])) for i in range(dfl.Nz())][0] #Hy = exp(iz0/my(k^2 - ky^2)^(1/2) - k)                  
            self.mask = Hx*Hy
        elif dfl.domains == 'ks':
            k = 2 * np.pi / self.xlamds
            Hx = [np.exp(1j * self.z0/self.mx * (np.sqrt(k ** 2 - k_x ** 2) - k)) for i in range(dfl.Nz())][0] 
            Hy = [np.exp(1j * self.z0/self.my * (np.sqrt(k ** 2 - k_y ** 2) - k)) for i in range(dfl.Nz())][0]          
            self.mask = Hx*Hy
        else: 
            raise ValueError('wrong field domain, domain must be ks or kf ')    
            
        return self.mask
    
class DriftMask(Mask):
    """
    TODO
    write documentation
    add logging
    """
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
'''
#optics elements check
dfl = RadiationField()
E_pohoton = 1239.8#200 #central photon energy [eV]
kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(101,101,11),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
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







