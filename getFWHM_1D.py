import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

def fit_and_print_info(x, y, plot=False):
	H, A, x0, sigma = gauss_fit(x, y)
	FWHM = 2.35482 * sigma
	#print('The offset of the gaussian baseline is', H)
	print('The center of the gaussian fit is', x0)
	print('The sigma of the gaussian fit is', np.abs(sigma))
	#print('The maximum intensity of the gaussian fit is', H + A)
	print('The Amplitude of the gaussian fit is', A)
	print('The FWHM of the gaussian fit is', np.abs(FWHM))

	if plot:
		plt.plot(x, y, 'ko', label='data')
		plt.plot(x, gauss(x, *gauss_fit(x, y)), '--r', label='fit')
		plt.legend()
		plt.title('Gaussian fit,  $f(x) = A e^{(-(x-x_0)^2/(2sigma^2))}$')
		plt.xlabel('Motor position')
		plt.ylabel('Intensity (A)')
		plt.show()