import matplotlib.pyplot as plt
import numpy as np
from scripts.model.kernel import Kernel
from scripts.model.parameters import Param

param = Param()
krnl = Kernel(param)
print(np.shape(krnl.weights)) # print dimensions of kernel
print(np.sum(krnl.weights[:,:,krnl.r_z])) # sum of center xy layer should be approximately one

# plot one slice
slice = krnl.weights[:,:,krnl.r_z]
# slice = np.transpose(krnl.weights[:,krnl.r_xy,:])

plt.figure()
plt.imshow(slice, cmap='gray', aspect='auto', origin ='lower', interpolation ='none', extent=[-7,7,-7,7])
plt.colorbar()
# plt.title(titletext)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# extent=[-7,7,-11,11] for xz view

# def sumsimulate(I_start,kernel,domain_dim):
#    return jnp.sum(simulate(I_start,kernel,domain_dim))

# grd = jax.grad(sumsimulate)
# print(grd(I_start,kernel,domain_dim))
