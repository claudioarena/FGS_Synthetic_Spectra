import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import convolve, convolve_fft
from astropy.convolution import AiryDisk2DKernel

simSize = 4
imw = 1280
imh = 200
# airyRad = 4

image = np.zeros((imh*simSize, imw*simSize))
image[int(simSize*imh/2.0), :] = 0.1
image[int(simSize*imh/2.0), int(imw*simSize/2.0)] = 1


## Generate high resultion Airy, and shift
airyRadius = 4 # in image pixels
airySimSize = 100 # 100 = allows shifts of 0.01 image pixels
maxShift = 2 # in image pixels

sim_min_dist = 8*airyRadius*airySimSize # minimum distance from airy center to border
sim_max_shift = maxShift*airySimSize
airySimRadius = airyRadius*airySimSize

airyKernelSize = 2*sim_min_dist + 4*sim_max_shift # Needed to contain all possible shifted arrays
airy_wide = AiryDisk2DKernel(radius=airySimRadius, x_size=airyKernelSize, y_size=airyKernelSize)
airy_wide_arr = airy_wide.array

# Now shift!
x_shift = 0.0 * airySimSize # image pixels
y_shift = 0.0 * airySimSize # image pixels

airyKernelShiftedSize = 2*sim_min_dist + 2*sim_max_shift # to always have enough space between airy and border
airyKerShiftedHalfSize = int(airyKernelShiftedSize / 2.0)
airyShiftedCenterX = int((airyKernelSize/2.0) + x_shift)
airyShiftedCenterY = int((airyKernelSize/2.0) + y_shift)

airy_shifted_array = airy_wide_arr[airyShiftedCenterY-airyKerShiftedHalfSize:airyShiftedCenterY+airyKerShiftedHalfSize,
                     airyShiftedCenterX-airyKerShiftedHalfSize:airyShiftedCenterX+airyKerShiftedHalfSize]

## Resample kernel
ratio = int(airySimSize / simSize)
final_array_size = int(airy_shifted_array.shape[1]/ratio)
airy_shifted_array = airy_shifted_array.reshape(final_array_size, ratio, final_array_size, ratio)  # (x_binned, x_bin_size, y_binned, y_bin_size)
airy_shifted_array = airy_shifted_array.sum(axis=(1,3))

#plt.figure()
#plt.imshow(airy_wide_arr)
#plt.figure()
#plt.imshow(airy_shifted_array)

z = convolve_fft(image, airy_shifted_array, allow_huge=True)

# a = np.arange(36).reshape(6, 6)
# Bin down by reshaping to 4-D array
z_sampled = z.reshape(imh, simSize, imw, simSize)  # (x_binned, x_bin_size, y_binned, y_bin_size)
z_sampled = z_sampled.mean(axis=(1,3))

plt.figure()
plt.imshow(z_sampled, cmap='gray', vmin=0, vmax=z.max())
plt.show()