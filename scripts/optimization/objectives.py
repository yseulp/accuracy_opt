"""Solver-independent base objectives for the optimization problems."""

import jax.numpy as jnp

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


def obj_fun_guven(I, target_E, C, kernel, domain_dim, param):
    """Guven base objective: minimize total outside energy."""
    del target_E
    E = simulate(I, kernel, domain_dim, param)
    return jnp.sum(E[C == 0])


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
