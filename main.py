import numpy as np
import matplotlib.pyplot as plt
import utility_methods, generate_spectra
import spectra_extract_methods as sp
import getFWHM_1D
import generate_convolved_spectra_image as gc
import pickle
import random
generate_spectra_image = True
gaussian_stddev = 1

simSize = 10
imw = 540 # account for some extra pixel to be cropped later on
imh = 100
maxShift = 40

if generate_spectra_image:
	ideal_spectra, kernel = gc.generate_spectra_image(gaussian_stddev=gaussian_stddev,
	                                            simSize=simSize, imw=imw, imh=imh, background_l=10, peak_l=150)
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
#spectra_PID_Off, _ = utility_methods.apply_spectra_shift(x_combined*17, x_combined*17, spectra, maxShift, simSize)
shift = np.linspace(-17, 18, 100)
const = np.linspace(0, 0, 100)
#x_PID_Off = np.random.rand(100)*16-8
#y_PID_Off = np.random.rand(100)*16-8
spectra_PID_Off, original_spectra = utility_methods.apply_spectra_shift(x_PID_Off*17, y_PID_Off*17, spectra, maxShift, simSize)
spectra_PID_Off_drift_x, _ = utility_methods.apply_spectra_shift(shift, const, spectra, maxShift, simSize)
spectra_PID_Off_drift_y, _ = utility_methods.apply_spectra_shift(const, shift, spectra, maxShift, simSize)

###### Add noise to final spectra and save ######
noise = 8.0
spectra_PID_On = utility_methods.add_noise(spectra_PID_On, biasNoise=noise)
spectra_PID_Off = utility_methods.add_noise(spectra_PID_Off, biasNoise=noise)
spectra_PID_Off_drift_x = utility_methods.add_noise(spectra_PID_Off_drift_x, biasNoise=noise)
spectra_PID_Off_drift_y = utility_methods.add_noise(spectra_PID_Off_drift_y, biasNoise=noise)
original_spectra = utility_methods.add_noise(original_spectra, biasNoise=noise)

def analyse_spectra(spectra, y_movement, apwidth, name, save=False):
	spectra = spectra[:, 20:-20]  # crop
	trace = sp.trace_spectra(spectra, show_extraction_area=False, nsteps=5)
	trace[:] = 50 + np.array(y_movement).mean()  # Fix problem with tracing
	f_red, _ = sp.extract_spectra(spectra, trace, show_extraction_area=False, apwidth=apwidth, skysep=13,
	                                skywidth=15)
	sp.display_spectra_image(spectra, save_pic=save, filename=name+'_image')
	f_red = sp.normalise_spectra_first_points(f_red, 50)
	x = np.arange(1, spectra.shape[1] + 1)
	# wav = 8.56438018e-02 * x + 5.32926548e+02
	wav = x * 0.0022 + 1.3
	sp.display_spectra(f_red, wav, save_pic=save, filename=name+'_spectra')
	sp.display_line_feature(f_red, wav, save_pic=save, filename=name+'_spectra_detail')
	l_x = wav[230:270]
	l_y = f_red[230:270]
	#plt.figure()
	#getFWHM_1D.fit_and_print_info(l_x, l_y, plot=True)


save = True
###### Analyse spectra ######
analyse_spectra(original_spectra, const, 3, 'Spectra-NoPointingErrors', save=save)
analyse_spectra(spectra_PID_On, y_PID_On, 3, 'Spectra-PIDOn', save=save)

###### Analyse spectra ######
#analyse_spectra(spectra_PID_Off, y_PID_Off, 18)
analyse_spectra(spectra_PID_Off_drift_x, const, 3, 'Spectra-XDrift', save=save)
analyse_spectra(spectra_PID_Off_drift_y, shift, 18, 'Spectra-YDrift', save=save)
