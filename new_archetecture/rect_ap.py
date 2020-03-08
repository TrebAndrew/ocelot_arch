import ocelot
from ocelot.common.globals import *
from ocelot import ocelog
from ocelot.common.ocelog import *
_logger = logging.getLogger(__name__) 

from ocelot.optics.wave import dfldomain_check, wigner_dfl, HeightProfile, generate_gaussian_dfl
from ocelot.gui.dfl_plot import plot_dfl, plot_wigner, plot_dfl_waistscan
import multiprocessing
nthread = multiprocessing.cpu_count()
from new_wave import*

E_pohoton = 1239.8#200 #central photon energy [eV]

kwargs={'xlamds':(h_eV_s * speed_of_light / E_pohoton), #[m] - central wavelength
        'rho':1.0e-4, 
        'shape':(201,201,1),             #(x,y,z) shape of field matrix (reversed) to dfl.fld
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

#dfl = RadiationField()
dfl = generate_gaussian_dfl(**kwargs);  #Gaussian beam defenition

appRect =  ApertureRectMask(lx=9e-4, ly=9e-4, cx=0, cy=0)
prop =  PropMask(z0=30)

appRect.apply(dfl)
prop.apply(dfl)

plot_dfl(dfl, phase=True, fig_name='after', domain='sf')























