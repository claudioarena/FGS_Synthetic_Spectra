import spectra_extract_methods as sp
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import numpy as np
from specutils.analysis import gaussian_sigma_width, gaussian_fwhm, fwhm
from specutils.analysis import snr_derived
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion

save_pic = False
ref_line_1 = SpectralRegion(585*u.micron, 590*u.micron)
ref_line_2 = SpectralRegion(590*u.micron, 596*u.micron)
ref_line_3 = SpectralRegion(596*u.micron, 603*u.micron)
ref_line_4 = SpectralRegion(628*u.micron, 634*u.micron)
ref_line = [ref_line_1, ref_line_2, ref_line_3, ref_line_4]

#### Read theref_line = SpectralRegion(585*u.micron, 590*u.micron)
im = Image.open("C:\\Users\\Claudio\\WinSysFiles\\Documents\\Python\\FGS_Breadboard\\data\\march2020\\20um_CFL_spectra_noPID\\image_9.tif")
img = np.zeros_like(im, dtype=np.float64)

for n in range(0, 18):
	filename = "C:\\Users\\Claudio\\WinSysFiles\\Documents\\Python\\FGS_Breadboard\\data\\march2020\\20um_CFL_spectra_noPID\\image_" + str(n) + ".tif"
	im = Image.open(filename)
	img = img + np.array(im)
	#print(img.max())


#Show data
#plt.figure()
#plt.imshow(img, origin='lower', aspect='auto', cmap=cm.Greys_r)
#plt.show()

f_red, trace, _ = sp.trace_spectra(img, show_extraction_area=False, nsteps=10, apwidth=10, skysep=25, skywidth=15)
f_red = sp.normalise_spectra(f_red)
wav = sp.w_calibrate(img)
sp.display_spectra(f_red, wav, save_pic)

wav_unit = wav * u.micron
spectrum_noPID = Spectrum1D(spectral_axis=wav_unit, flux=f_red * u.dimensionless_unscaled)
print(gaussian_sigma_width(spectrum=spectrum_noPID, regions=ref_line))
print(gaussian_fwhm(spectrum=spectrum_noPID, regions=ref_line))
print(fwhm(spectrum=spectrum_noPID, regions=ref_line))
print(snr_derived(spectrum=spectrum_noPID))
print(snr_derived(spectrum=spectrum_noPID, region=ref_line))


#### Read the data. - PID ON
im = Image.open("C:\\Users\\Claudio\\WinSysFiles\\Documents\\Python\\FGS_Breadboard\\data\\march2020\\20um_CFL_spectra_PID_On\\image_9.tif")
img = np.zeros_like(im, dtype=np.float64)

for n in range(0, 18):
	filename = "C:\\Users\\Claudio\\WinSysFiles\\Documents\\Python\\FGS_Breadboard\\data\\march2020\\20um_CFL_spectra_PID_On\\image_" + str(n) + ".tif"
	im = Image.open(filename)
	img = img + np.array(im)
	#print(img.max())

f_red, trace, _ = sp.trace_spectra(img, show_extraction_area=False, nsteps=10, apwidth=10, skysep=25, skywidth=15)
f_red = sp.normalise_spectra(f_red)
wav = sp.w_calibrate(img)
sp.display_spectra(f_red, wav, save_pic)

wav_unit = wav * u.micron
spectrum_PID = Spectrum1D(spectral_axis=wav_unit, flux=f_red * u.dimensionless_unscaled)
print(gaussian_sigma_width(spectrum=spectrum_PID, regions=ref_line))
print(gaussian_fwhm(spectrum=spectrum_PID, regions=ref_line))
print(fwhm(spectrum=spectrum_PID, regions=ref_line))
print(snr_derived(spectrum=spectrum_PID))
print(snr_derived(spectrum=spectrum_PID, region=ref_line))

#plt.show()