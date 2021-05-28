import generate_spectra
from astropy.convolution import convolve, convolve_fft
import matplotlib.pyplot as plt
import pickle
import numpy as np


#maxShift only needed if we plan to move the gaussian center in the kernel
def generate_spectra_image(gaussian_stddev=3, simSize=10, imw=1280, imh=200, maxShift=0, background_l=10, peak_l=30):
	###### Load 1D spectrum ######
	ideal_spectra = generate_spectra.generate_peaked_spectra(imw*simSize, imh*simSize, background_level=background_l, peak_level=peak_l, peak_dist=simSize*gaussian_stddev)

	###### generate syntethic 2D spectra ######
	#airyKernel = generate_spectra.generate_airy_kernel(airyRadius=airyR, airySimSize=simSize, maxShift=maxShift)
	kernel = generate_spectra.generate_gaussian_kernel(x_stddev=gaussian_stddev, SimSize=simSize, maxShift=maxShift)

	return ideal_spectra, kernel


def display_spectra_image(spectra):
	plt.figure()
	plt.title('single spectra')
	plt.imshow(spectra, cmap='gray', vmin=0, vmax=spectra.max())
	plt.show()

def convolve_mine(ideal_spectra, kernel, save=True):
	ideal_spectra_h = ideal_spectra.shape[0]
	spectra_pos_y = ideal_spectra_h / 2
	kernel_h = kernel.shape[0]
	ideal_spectra_subarray = ideal_spectra[int(spectra_pos_y - (kernel_h * 1.2)):int(spectra_pos_y + (kernel_h * 1.2)), :]

	spectra_sub = convolve(ideal_spectra_subarray, kernel)
	spectra = np.zeros_like(ideal_spectra)

	spectra[int(spectra_pos_y - (kernel_h * 1.2)):int(spectra_pos_y + (kernel_h * 1.2)), :] = spectra_sub
	#make the spectra peak at 200
	spectra = spectra * (200/spectra.max())

	#crop sides after convolve
	#spectra = spectra[:, :]

	#if save:
	#	pickle_spectra_image(spectra, kernel, ideal_spectra)

	return spectra


def pickle_spectra_image(spectra, kernel, ideal_spectra):
	filename = 'spectra_image_simSize10_stdev3'
	outfile = open(filename, 'wb')
	pickle.dump(spectra, outfile)
	pickle.dump(kernel, outfile)
	pickle.dump(ideal_spectra, outfile)
	outfile.close()