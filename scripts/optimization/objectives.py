"""Solver-independent base objectives for the optimization problems."""

import jax.numpy as jnp
from jax import lax

from scripts.model import indexing
from scripts.model.simulation import simulate


def obj_fun_1(I, target_E, C, kernel, domain_dim, param):
    """Squared energy-target error."""
    del C
    E = simulate(I, kernel, domain_dim, param)
    return jnp.sum((E - target_E) ** 2)


def obj_fun_2(I, target_E, C, kernel, domain_dim, param):
    """Original soft objective measuring total cure-threshold violation."""
    del target_E
    E = simulate(I, kernel, domain_dim, param)
    return jnp.sum(jnp.abs(jnp.minimum(0, E[C == 1] - param.E_crit))) + jnp.sum(
        jnp.abs(jnp.maximum(0, E[C == 0] - param.E_crit))
    )


def guven_boundary_mask(C, domain_dim):
    """Return the one-voxel-wide exterior XY boundary around target voxels.

    The 3x3x1 neighborhood includes diagonal XY neighbors but never expands
    into adjacent Z layers. The returned boolean mask has the same shape as C.
    """
    target = indexing.to_3d(jnp.asarray(C) == 1, domain_dim)
    neighboring_target = lax.reduce_window(
        target.astype(jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        lax.max,
        window_dimensions=(3, 3, 1),
        window_strides=(1, 1, 1),
        padding="SAME",
    ).astype(bool)
    boundary = neighboring_target & ~target
    return jnp.reshape(boundary, jnp.shape(C))


def obj_fun_guven(I, target_E, C, kernel, domain_dim, param):
    """Guven base objective: minimize energy on the exterior XY boundary."""
    del target_E
    E = simulate(I, kernel, domain_dim, param)
    boundary = guven_boundary_mask(C, domain_dim)
    return jnp.sum(E * boundary.astype(E.dtype))


def obj_fun_wang(I, target_E, C, kernel, domain_dim, param):
    """Wang base objective: L1 energy-target error."""
    del C
    E = simulate(I, kernel, domain_dim, param)
    return jnp.sum(jnp.abs(E - target_E))


def obj_fun_reverse_guven(I, target_E, C, kernel, domain_dim, param):
    """Reverse-Guven base objective: maximize total inside energy."""
    del target_E
    E = simulate(I, kernel, domain_dim, param)
    return -jnp.sum(E[C == 1])
