from astropy.convolution import convolve, convolve_fft
from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel
import numpy as np

def generate_airy_kernel(airyRadius=4, airySimSize=100, maxShift=2):
	sim_min_dist = 8 * airyRadius * airySimSize  # minimum distance from airy center to border
	sim_max_shift = maxShift * airySimSize
	airy_sim_radius = airyRadius * airySimSize

	airy_kernel_size = 2 * sim_min_dist + 4 * sim_max_shift + 1# Needed to contain all possible shifted arrays
	airy = AiryDisk2DKernel(radius=airy_sim_radius, x_size=airy_kernel_size, y_size=airy_kernel_size)
	airy_arr = airy.array

	return airy_arr


def generate_gaussian_kernel(x_stddev=4, SimSize=10, maxShift=2):
	sim_min_dist = 8 * x_stddev * SimSize  # minimum distance from airy center to border
	sim_max_shift = maxShift * SimSize
	x_sim_stddev = x_stddev * SimSize

	kernel_size = 2 * sim_min_dist + 4 * sim_max_shift + 1 # Needed to contain all possible shifted arrays
	if maxShift is 0:
		kernel_size = sim_min_dist+1

	gaussian = Gaussian2DKernel(x_stddev=x_sim_stddev, x_size=kernel_size, y_size=kernel_size)
	gaussian_array = gaussian.array

	return gaussian_array

def generate_peaked_spectra(length=1024, height=200, background_level=20, peak_level=250, peak_dist=20):
	image = np.zeros((height, length))
	image[int(height / 2.0), :] = background_level
	peak_wavelength = length / 6 ##generate 5 peaks

	image[int(height / 2.0), int(peak_wavelength)] = peak_level
	image[int(height / 2.0), int(peak_wavelength*2)] = peak_level
	if peak_dist is not 0:
		image[int(height / 2.0), int(peak_wavelength*2+peak_dist*2.7)] = peak_level

	image[int(height / 2.0), int(peak_wavelength)*3] = peak_level
	image[int(height / 2.0), int(peak_wavelength)*4] = peak_level
	image[int(height / 2.0), int(peak_wavelength)*5] = peak_level

	return image