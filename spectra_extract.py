import spectra_extract_methods as sp
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np

save_pic = False
#### Read theref_line = SpectralRegion(585*u.micron, 590*u.micron)
im = Image.open("C:\\Users\\Claudio\\WinSysFiles\\Documents\\Python\\FGS_Breadboard\\data\\march2020\\20um_CFL_spectra_noPID\\image_9.tif")
img = np.array(im, dtype=np.float64)

#Show data
#plt.figure()
#plt.imshow(img, origin='lower', aspect='auto', cmap=cm.Greys_r)
#plt.show()

f_red, trace, _ = sp.trace_spectra(img, show_extraction_area=False, nsteps=10, apwidth=10, skysep=25, skywidth=15)
f_red = sp.normalise_spectra(f_red)
wav = sp.w_calibrate(img)
sp.display_spectra(f_red, wav, save_pic)

#plt.show()