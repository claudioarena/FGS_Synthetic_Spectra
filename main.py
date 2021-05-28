import numpy as np
import matplotlib.pyplot as plt
import utility_methods, generate_spectra
import spectra_extract_methods as sp
import getFWHM_1D
import generate_convolved_spectra_image as gc
import pickle
import random
generate_spectra_image = True
gaussian_stddev = 3

simSize = 4
imw = 1280
imh = 200
maxShift = 20

if generate_spectra_image:
	ideal_spectra, kernel = gc.generate_spectra_image(gaussian_stddev=gaussian_stddev,
	                                            simSize=simSize, imw=imw, imh=imh, background_l=20, peak_l=100)
	spectra = gc.convolve_mine(ideal_spectra, kernel, save=False)
else:
	f = open("spectra_image_simSize10_stdev3.pkl", 'rb')
	spectra = pickle.load(f)
	f.close()

gc.display_spectra_image(spectra)

# resize, to allow for small pixel shifts
#utility_methods.upsample_image(spectra, 50/simSize)
#simSize = 50

###### Load PSF movements ######
x_PID_On, y_PID_On, x_PID_Off, y_PID_Off = utility_methods.unpickle_movements()

###### Shift and sum spectra ######
spectra_PID_On, _ = utility_methods.apply_spectra_shift(x_PID_On, y_PID_On, spectra, maxShift, simSize)
#x_combined = np.concatenate([x_PID_Off*4, y_PID_Off*4])
#spectra_PID_Off, _ = utility_methods.apply_spectra_shift(x_combined*4, x_combined*4, spectra, maxShift, simSize)
x_combined = np.linspace(-5, 5, 100)
x_off = np.random.rand(100)*16-8
y_off = np.random.rand(100)*16-8
spectra_PID_Off, _ = utility_methods.apply_spectra_shift(x_off, y_off, spectra, maxShift, simSize)

###### Add noise to final spectra and save ######
spectra_PID_On = utility_methods.add_noise(spectra_PID_On, biasNoise=8.0)
spectra_PID_Off = utility_methods.add_noise(spectra_PID_Off, biasNoise=8.0)

###### Analyse spectra ######
spectra_PID_On = spectra_PID_On[:, 20:-20] # crop
trace_1 = sp.trace_spectra(spectra_PID_On, show_extraction_area=False, nsteps=5)
trace_1[:] = 100 + np.array(y_PID_On).mean() #Fix problem with tracing
f_red_1, _ = sp.extract_spectra(spectra_PID_On, trace_1, show_extraction_area=True, apwidth=4, skysep=25, skywidth=15)
f_red_1 = sp.normalise_spectra_first_points(f_red_1, 150)
x = np.arange(1, spectra_PID_On.shape[1] + 1)
#wav = 8.56438018e-02 * x + 5.32926548e+02
wav_1 = x + 5.32926548e+02
sp.display_spectra(f_red_1, wav_1, save_pic=False)
l_x_1 = wav_1[550:700]
l_y_1 = f_red_1[550:700]
plt.figure()
getFWHM_1D.fit_and_print_info(l_x_1, l_y_1,plot=True)

###### Analyse spectra ######
spectra_PID_Off = spectra_PID_Off[:, 20:-20] # crop
trace = sp.trace_spectra(spectra_PID_Off, show_extraction_area=False, nsteps=5)
trace[:] = 100 + np.array(y_PID_Off).mean() #Fix problem with tracing
f_red, _ = sp.extract_spectra(spectra_PID_Off, trace, show_extraction_area=True, apwidth=4+8, skysep=25, skywidth=15)
f_red = sp.normalise_spectra_first_points(f_red, 150)
x = np.arange(1, spectra_PID_Off.shape[1] + 1)
#wav = 8.56438018e-02 * x + 5.32926548e+02
wav = x + 5.32926548e+02
sp.display_spectra(f_red, wav, save_pic=False)
l_x = wav[550:700]
l_y = f_red[550:700]
plt.figure()
getFWHM_1D.fit_and_print_info(l_x,l_y,plot=True)