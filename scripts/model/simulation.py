from scripts.model import indexing
import jax.numpy as jnp
from jax import lax

def simulate(mask_I,kernel,domain_dim,param):

    I_3D = indexing.to_3d(mask_I,domain_dim) # convert to 3D matrix

    # JAX conv expects (N, C, D, H, W), where N = batch size, C number of input channels, both not relevant for this example
    y = lax.conv_general_dilated(
        rhs=kernel.weights[jnp.newaxis, jnp.newaxis, ...],
        lhs=I_3D[jnp.newaxis, jnp.newaxis, ...], 
        window_strides=(1, 1, 1), # convolution loop over every pixel
        padding='SAME', #Padding: "constant" with zeros, output has same dimension as input
    )
      
    # reformat to 1D, remove batch & channel dims, multipy with layer time
    energy_exposure = indexing.flatten_3d(y[0, 0])* param.exposure_time

    return energy_exposure

def cured(energy, param):
    return (energy >= param.E_crit)
