from ocelot.utils.xfel_utils import *
from ocelot.gui.dfl_plot import *
from ocelot.rad.optics_elements import *
from ocelot.rad.optics_line import *
from ocelot.rad.propagation import *


dfl = generate_gaussian_dfl(shape=(151, 151, 1))


d = FreeSpace(l=10, mx=2, my=2)
l = ThinLens(fx=10)
m = Mirror()
a = ApertureRect(lx=100e-6, ly=400e-6)
a2 = ApertureRect(lx=400e-6, ly=100e-6)
d2 = FreeSpace(l=10, mx=0.5, my=0.5)
line = (d, a, a2 , l, d2)
#line = ()


lat = OpticsLine(line)

#a.mask.apply(dfl)
#estimate_masks(lat, dfl)
#mask_dl = a.mask * a2.mask
#mask_dl.apply(dfl)
#plot_dfl(dfl)

dfl = propagate(lat, dfl)

plot_dfl(dfl)




