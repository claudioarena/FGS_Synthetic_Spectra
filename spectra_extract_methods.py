import pydis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import EngFormatter
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from PIL import Image, ImageOps
import PIL

apwidth = 10
skysep = 25
skywidth = 15

#Bias calibrate. Might not be needed.
#bias = pydis.biascombine('rbias.lis', trim=True)
#data = (img_spectra - bias)

#Wavelength cal
#HeNeAr_file = 'cal.fit'
#dataRegion = img[470:520, 1:img.shape[0]+1]
#wfit = pydis.HeNeAr_fit(HeNeAr_file, trim=False, interac=True, mode='poly', fit_order=2)

def showExctractionArea(img_spectra, trace, apwidth=10, skysep=25, skywidth=15):
    xbins = np.arange(img_spectra.shape[1])

    plt.figure()
    plt.imshow(img_spectra, origin='lower', aspect='auto', cmap=cm.Greys_r)

    # the trace
    plt.plot(xbins, trace, 'b', lw=1)

    # the aperture
    plt.plot(xbins, trace-apwidth, 'r', lw=1)
    plt.plot(xbins, trace+apwidth, 'r', lw=1)

    # the sky regions
    plt.plot(xbins, trace-apwidth-skysep, 'g', lw=1)
    plt.plot(xbins, trace-apwidth-skysep-skywidth, 'g', lw=1)
    plt.plot(xbins, trace+apwidth+skysep, 'g', lw=1)
    plt.plot(xbins, trace+apwidth+skysep+skywidth, 'g', lw=1)

    plt.title('(with trace, aperture, and sky regions)')
    plt.draw()


def trace_spectra(image, show_extraction_area=False, nsteps=10):
	# trace the science image
	trace = pydis.ap_trace(image, nsteps=nsteps, interac=False, display=show_extraction_area)
	return trace


def extract_spectra(image, trace, show_extraction_area=False, apwidth=10, skysep=25, skywidth=15):
	if show_extraction_area:
		showExctractionArea(image, trace, apwidth, skysep, skywidth)
	#extract
	ext_spec, sky, fluxerr = pydis.ap_extract(image, trace, apwidth=apwidth, skysep=skysep, skywidth=skywidth, skydeg=0)
	flux_red = (ext_spec - sky)  # the reduced object
	return flux_red, fluxerr


def normalise_spectra(flux_red):
	#normalise
	flux_red = (flux_red - np.min(flux_red))
	flux_red = (flux_red / np.max(flux_red))
	return flux_red

def normalise_spectra_first_points(flux_red, n):
	#normalise
	flux_red = (flux_red - flux_red[0:n].mean())
	flux_red = (flux_red / np.max(flux_red))
	return flux_red

def w_calibrate(image):
	# Wavelenght calibrate
	# wfinal = pydis.mapwavelength(trace, wfit, mode='poly')

	# From rough wav cal: y = -3E-05x2 + 0.1283x + 524.46R = 0.995
	x = np.arange(1, image.shape[1] + 1)
	wav = -7.04458602e-07 * x * x + 8.56438018e-02 * x + 5.32926548e+02
	# wav = 0.0876*x + 529.85

	return wav


def display_spectra(f_red, wav, save_pic=False, filename='_spectra', title='Spectra'):
	# Display result
	# plt.figure()
	# plt.plot(wav, f_red)
	# #plt.show(block=False)
	# #plt.pause(0.1)
	# # plt.errorbar(flux_red, yerr=fluxerr)
	# plt.xlabel('Wavelength [??m]')
	# plt.ylabel('Normalised flux')
	# plt.title(title)
	# # plot within percentile limits
	# #plt.ylim(0, 1700)
	# #plt.ion()
	# plt.show()

	if save_pic:
		matplotlib.use("pgf")
		matplotlib.rcParams.update({
			"pgf.texsystem": "pdflatex",
			'font.family': 'serif',
			'text.usetex': True,
			'pgf.rcfonts': False,
		})

	# set reasonable figsize for 1-column figures
	# fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
	fig = plt.figure(figsize=(145 / 25.4, ((237-58-58)/2) / 25.4)) #A4=210*297. Print area: 237 h, 145 w
	# 500x100 image is 145x29 mm. So two are 58 mm tall. 58 for two captions too
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(wav, f_red, color='blue', linewidth=1)
	plt.show(block=False)
	ax.set_xlim([1.3, 2.4])
	ax.set_xlabel('Wavelength [$\mu$m]')
	ax.set_ylabel('Normalised flux')
	#ax.set_title(title)
	ax.xaxis.set_ticks(np.arange(1.3, 2.4, 0.2))
	ax.xaxis.grid(False, which='minor')
	ax.xaxis.set_major_locator(MultipleLocator(0.2))
	ax.xaxis.set_minor_locator(MultipleLocator(0.05))
	plt.tight_layout()
	plt.show()

	# when using this interface, we need to explicitly call the draw routine
	plt.draw()
	# plt.figure()
	plt.show()

	if save_pic:
		fn = filename + '.pgf'
		plt.savefig(fn)
		plt.savefig(filename + '.png', dpi=600)


def display_line_feature(f_red, wav, save_pic=False, filename='_spectra', title='Spectra'):
	if save_pic:
		matplotlib.use("pgf")
		matplotlib.rcParams.update({
			"pgf.texsystem": "pdflatex",
			'font.family': 'serif',
			'text.usetex': True,
			'pgf.rcfonts': False,
		})

	### PLOT LINE DETAIL
	x_start = 1.6 - 0.045
	x_end = 1.72 + 0.045
	# set reasonable figsize for 1-column figures
	# fig, axs = plt.subplots(1, 1, constrained_layout=True, figsize=(6.3, (8.5776-1)/3))
	fig = plt.figure(figsize=(((237-10) / 2) / 25.4, (((145-29)/2) / 25.4))) # A4=210*297. Print area: 237 h, 145 w
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(wav, f_red, color='blue', linewidth=1)
	plt.show(block=False)
	ax.set_xlim([x_start,x_end])
	ax.set_xlabel('Wavelength [$\mu$m]')
	ax.set_ylabel('Normalised flux')
	#ax.set_title(title)
	ax.xaxis.set_ticks(np.arange(x_start, x_end, 0.025))
	ax.xaxis.grid(False, which='minor')
	ax.xaxis.set_major_locator(MultipleLocator(0.025))
	ax.xaxis.set_minor_locator(MultipleLocator(0.005))
	plt.tight_layout()
	plt.show()

	# when using this interface, we need to explicitly call the draw routine
	plt.draw()
	# plt.figure()
	plt.show()

	if save_pic:
		fn = filename + '.pgf'
		plt.savefig(fn)
		plt.savefig(filename + '.png', dpi=600)


def display_spectra_image(spectra, save_pic=False, filename='_image'):
	PIL_image = Image.fromarray(np.uint8(spectra))
	inverted_image = PIL.ImageOps.invert(PIL_image)
	inverted_image.show()
	gray_image = ImageOps.grayscale(inverted_image)

	if save_pic:
		gray_image.save(filename+'.png')