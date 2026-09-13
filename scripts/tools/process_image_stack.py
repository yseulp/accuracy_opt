import numpy as np
import imageio
import glob

from scipy import ndimage 
from scripts.model.kernel import Kernel
from scripts.model.parameters import Param

# import sparse does not work 

# Path pattern to your PNG files
file_list = sorted(glob.glob("data/eifel_tower/*.png"))

# Read and normalize each image
print("reading images...")
images = [imageio.v2.imread(fname).astype(np.float32) / 255.0 for fname in file_list]

# Stack into a 3D numpy array (N_images, H, W)
print("stack into 3D array...")
volume = np.stack(images, axis=0)

print("Array shape:", volume.shape)
print("Data type:", volume.dtype)

# --------------------------------------------------------------------

p = Param()
influence_kernel = Kernel(p)

print('calculate the convolution')
# mode=constant sets kernel weights outside the boundary to zero.
energy_exposure_3D = ndimage.convolve(
    volume,
    influence_kernel.weights,
    mode='constant',
    cval=0.0,
    origin=[0, 0, -(influence_kernel.r_z // 2)],
) * p.exposure_time

print('preparing sparse array')
coords = np.argwhere(energy_exposure_3D != 0)
values = energy_exposure_3D[energy_exposure_3D > 1.0e-8]

# Save sparse to disk
print('saving to disk')
np.savez_compressed("data/energy_exposure.npz", coords=coords, values=values, shape=energy_exposure_3D.shape)
