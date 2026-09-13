"""Legacy convenience wrappers around the canonical solver dispatch."""

from __future__ import annotations

from copy import copy

import numpy as np

from scripts.optimization.formulations import build_inside_energy_matrix
from scripts.optimization.solvers import run_solver
from scripts.model.simulation import simulate


def optimize_guven_scipy(I_start, T, C, kernel, domain_dim, param):
    """Run the constrained Guven problem with SLSQP."""
    result = _run_guven(
        "slsqp",
        I_start,
        T,
        C,
        kernel,
        domain_dim,
        param,
        solver_parameters={"max_iterations": 100, "ftol": 1e-6},
    )
    return np.asarray(result.I_opt)


def optimize_guven_trust_constr(I_start, T, C, kernel, domain_dim, param):
    """Run the constrained Guven problem with trust-constr."""
    result = _run_guven(
        "trust_constr",
        I_start,
        T,
        C,
        kernel,
        domain_dim,
        param,
        solver_parameters={"max_iterations": 250},
    )
    return np.asarray(result.I_opt)


def optimize_guven_linprog(
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    *,
    safety_margin=1e-4,
    verify_matrix=False,
):
    """Run the constrained Guven problem as a linear program with HiGHS."""
    if verify_matrix:
        _verify_inside_energy_matrix(C, kernel, domain_dim, param)
    adjusted_param = copy(param)
    adjusted_param.E_crit = float(param.E_crit) + float(safety_margin)
    result = _run_guven(
        "linprog_highs",
        I_start,
        T,
        C,
        kernel,
        domain_dim,
        adjusted_param,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.I_opt)


def _run_guven(
    solver,
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    *,
    problem_parameters=None,
    solver_parameters=None,
):
    return run_solver(
        solver,
        "guven",
        I_start=I_start,
        T=T,
        C=C,
        kernel=kernel,
        domain_dim=domain_dim,
        param=copy(param),
        problem_parameters=problem_parameters or {},
        solver_parameters=solver_parameters or {},
    )


def _verify_inside_energy_matrix(C, kernel, domain_dim, param) -> None:
    matrix, target_indices = build_inside_energy_matrix(
        C, kernel, domain_dim, param
    )
    rng = np.random.default_rng(42)
    for _ in range(5):
        intensity = rng.uniform(0.0, param.I_max, size=matrix.shape[1])
        matrix_energy = np.asarray(matrix @ intensity, dtype=np.float64)
        simulated_energy = np.asarray(
            simulate(intensity, kernel, domain_dim, param), dtype=np.float64
        )[target_indices]
        if not np.allclose(matrix_energy, simulated_energy, rtol=2e-4, atol=1e-3):
            raise RuntimeError("A_inside verification failed.")
