import numpy as np
import matplotlib.pyplot as plt
import pickle
import PIL
from skimage.util import random_noise

def make_zero_padded_spectra(spectra, maxShift): #max shift in sim pixels
	padded_spectra_height = spectra.shape[0] + 2 * maxShift
	padded_spectra_width = spectra.shape[1] + 2 * maxShift

	padded_spectra = np.zeros((padded_spectra_height, padded_spectra_width))
	return padded_spectra


def shift_and_sum(x_shift, y_shift, result_image, image_to_sum): #shifts in sim pixels
	shifted_cx = int((result_image.shape[1] / 2.0) + x_shift)
	shifted_cy = int((result_image.shape[0] / 2.0) + y_shift)

	half_height = int(image_to_sum.shape[0]/2.0)
	half_width = int(image_to_sum.shape[1]/2.0)

	result_copy = np.zeros_like(result_image)
	result_copy[shifted_cy - half_height:shifted_cy + half_height,
	shifted_cx - half_width:shifted_cx + half_width] = image_to_sum
	result_image = result_image + result_copy

	return result_image


def apply_spectra_shift(x, y, spectra, maxShift, simSize, debug=False):
	padded_spectra = make_zero_padded_spectra(spectra, maxShift * simSize)

	for i in range(0, len(x)):
		padded_spectra = shift_and_sum(x[i] * simSize, y[i] * simSize, padded_spectra, spectra)
	# plt.figure()
	# plt.imshow(padded_spectra[400:440, 2540:2620])

	# crop and divide
	summed_spectra = padded_spectra[maxShift * simSize:-maxShift * simSize, maxShift * simSize:-maxShift * simSize]
	summed_spectra = (summed_spectra / len(x))

	if debug:
		plt.figure()
		plt.title('summed spectra')
		plt.imshow(summed_spectra, cmap='gray', vmin=0, vmax=summed_spectra.max())

	spectra_final = bin_image(summed_spectra, simSize)
	original_spectra_final = bin_image(spectra, simSize)

	if debug:
		plt.figure()
		plt.title('final spectra')
		plt.imshow(spectra_final, cmap='gray', vmin=0, vmax=spectra_final.max())

		plt.figure()
		plt.title('final original spectra')
		plt.imshow(original_spectra_final, cmap='gray', vmin=0, vmax=spectra_final.max())

	return spectra_final, original_spectra_final


def bin_image(image, bin_size): # Make sure image size is divisible by bin_size
	binned_size_x = int(image.shape[0] / bin_size)
	binned_size_y = int(image.shape[1] / bin_size)
	image_binned = image.reshape(binned_size_x, bin_size, binned_size_y, bin_size)
	image_binned = image_binned.mean(axis=(1, 3))
	return image_binned


def unpickle_movements():
	f = open("PID_On_Off_movements.pkl", 'rb')
	x_PID_On = pickle.load(f)
	y_PID_On = pickle.load(f)
	x_PID_Off = pickle.load(f)
	y_PID_Off = pickle.load(f)
	f.close()

	x_PID_On = np.mean(x_PID_On[0:35200].reshape(-1, 400), axis=1)
	y_PID_On = np.mean(y_PID_On[0:35200].reshape(-1, 400), axis=1)
	x_PID_Off = np.mean(x_PID_Off[0:59600].reshape(-1, 400), axis=1)
	y_PID_Off = np.mean(y_PID_Off[0:59600].reshape(-1, 400), axis=1)

	return x_PID_On, y_PID_On, x_PID_Off, y_PID_Off


def upsample_image(image, upsample_factor):
	PIL_image = PIL.Image.fromarray(image)
	image_resized = PIL_image.resize((int(image.shape[1]*upsample_factor), int(image.shape[0]*upsample_factor)), resample=PIL.Image.BILINEAR)
	image = np.array(image_resized)
	return image_resized


def add_noise(image, biasNoise=3):
	# bias
	noise = np.random.poisson(lam=biasNoise, size=(image.shape))
	image = image + noise

	# poisson
	image = random_noise(image, mode="poisson", clip=False)
	image = image.round(0)
	return image
