"""Adapters from mathematical problems to solver-ready representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse

from scripts.model.simulation import simulate
from scripts.optimization.objectives import guven_boundary_mask


@dataclass(frozen=True)
class ObjectiveFormulation:
    objective: Callable
    bounds: list[tuple[float, float]]


@dataclass(frozen=True)
class FunctionalConstraintSpec:
    function: Callable
    jacobian: Callable
    lower_bound: float | np.ndarray
    upper_bound: float | np.ndarray


@dataclass(frozen=True)
class LinearConstraintSpec:
    matrix: Any
    lower_bound: float | np.ndarray
    upper_bound: float | np.ndarray


@dataclass(frozen=True)
class ConstrainedFormulation:
    objective: Callable
    objective_grad: Callable
    constraints: tuple[FunctionalConstraintSpec | LinearConstraintSpec, ...]
    bounds: list[tuple[float, float]]


@dataclass(frozen=True)
class LinearFormulation:
    c: np.ndarray
    A_ub: Any | None
    b_ub: np.ndarray | None
    A_eq: Any | None
    b_eq: np.ndarray | None
    bounds: list[tuple[float | None, float | None]]
    intensity_size: int
    initial_point: Callable[[np.ndarray], np.ndarray]

    def extract_intensity(self, solution: np.ndarray) -> np.ndarray:
        return np.asarray(solution[: self.intensity_size], dtype=np.float64)


@dataclass(frozen=True)
class PenaltyPolicy:
    default_weight: float
    power: int
    smooth: bool


@dataclass(frozen=True)
class FormulationDescriptor:
    name: str
    result_label: str
    constraint_handling: str
    smooth: bool


@dataclass(frozen=True)
class FormulationAdapter:
    name: str
    result_label: str
    constraint_handling: str | Callable[[Any], str]
    supports: Callable[[Any], bool]
    is_smooth: Callable[[Any], bool]
    build: Callable

    def describe(self, problem) -> FormulationDescriptor:
        constraint_handling = self.constraint_handling
        if callable(constraint_handling):
            constraint_handling = constraint_handling(problem)
        return FormulationDescriptor(
            name=self.name,
            result_label=self.result_label,
            constraint_handling=constraint_handling,
            smooth=self.is_smooth(problem),
        )


PENALTY_POLICIES: dict[str, PenaltyPolicy] = {
    "guven": PenaltyPolicy(default_weight=100.0, power=2, smooth=True),
    "wang": PenaltyPolicy(default_weight=10.0, power=1, smooth=False),
    "reverse_guven": PenaltyPolicy(default_weight=10.0, power=2, smooth=True),
}


def resolve_penalty_parameters(
    problem_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if problem_name not in PENALTY_POLICIES:
        if parameters:
            raise ValueError(
                f"Problem {problem_name!r} has no exterior-penalty adapter."
            )
        return {}
    unknown = set(parameters) - {"penalty_weight"}
    if unknown:
        raise ValueError(
            f"Penalty adapter for {problem_name!r} does not use parameter(s): "
            + ", ".join(sorted(unknown))
        )
    weight = parameters.get(
        "penalty_weight",
        PENALTY_POLICIES[problem_name].default_weight,
    )
    if isinstance(weight, bool):
        raise ValueError("penalty_weight must be a non-negative finite number.")
    try:
        weight = float(weight)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "penalty_weight must be a non-negative finite number."
        ) from error
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("penalty_weight must be a non-negative finite number.")
    return {"penalty_weight": weight}


def _build_objective(
    problem,
    *,
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    problem_parameters,
    penalty_parameters,
):
    del penalty_parameters
    return ObjectiveFormulation(
        objective=lambda I: problem.objective(
            I, T, C, kernel, domain_dim, param, problem_parameters
        ),
        bounds=problem.bounds(len(I_start), param),
    )


def _build_penalty_objective(
    problem,
    *,
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    problem_parameters,
    penalty_parameters,
):
    policy = PENALTY_POLICIES[problem.name]
    weight = penalty_parameters["penalty_weight"]

    def objective(I):
        base_value = problem.objective(
            I, T, C, kernel, domain_dim, param, problem_parameters
        )
        residuals = problem.constraint_residuals(
            I, T, C, kernel, domain_dim, param, problem_parameters
        )
        violation = jnp.maximum(0.0, -residuals)
        return base_value + weight * jnp.sum(violation**policy.power)

    return ObjectiveFormulation(
        objective=objective,
        bounds=problem.bounds(len(I_start), param),
    )


def _build_constrained(
    problem,
    *,
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    problem_parameters,
    penalty_parameters,
):
    del penalty_parameters

    def objective_jax(I):
        return problem.objective(
            I, T, C, kernel, domain_dim, param, problem_parameters
        )

    def constraints_jax(I):
        return problem.constraint_residuals(
            I, T, C, kernel, domain_dim, param, problem_parameters
        )

    objective_value_and_grad = jax.value_and_grad(objective_jax)
    constraint_jacobian = jax.jacrev(constraints_jax)

    def objective(I):
        return float(objective_value_and_grad(jnp.asarray(I))[0])

    def objective_grad(I):
        return np.asarray(
            objective_value_and_grad(jnp.asarray(I))[1], dtype=np.float64
        )

    def constraints(I):
        return np.asarray(constraints_jax(jnp.asarray(I)), dtype=np.float64)

    def constraints_jac(I):
        return np.asarray(
            constraint_jacobian(jnp.asarray(I)), dtype=np.float64
        )

    return ConstrainedFormulation(
        objective=objective,
        objective_grad=objective_grad,
        constraints=(
            FunctionalConstraintSpec(
                function=constraints,
                jacobian=constraints_jac,
                lower_bound=0.0,
                upper_bound=np.inf,
            ),
        ),
        bounds=problem.bounds(len(I_start), param),
    )


def _flat_to_xyz(flat_idx, dims):
    dim_x, dim_y, dim_z = dims
    x = flat_idx // (dim_y * dim_z)
    rem_yz = flat_idx % (dim_y * dim_z)
    return x, rem_yz // dim_z, rem_yz % dim_z


def _xyz_to_flat(x, y, z, dims):
    _, dim_y, dim_z = dims
    return x * (dim_y * dim_z) + y * dim_z + z


def _kernel_support(kernel):
    weights = np.asarray(jnp.asarray(kernel.weights), dtype=np.float64)
    return [
        (
            int(ax) - kernel.r_xy,
            int(ay) - kernel.r_xy,
            int(az) - kernel.r_z,
            float(weights[ax, ay, az]),
        )
        for ax, ay, az in np.argwhere(weights != 0)
    ]


def _build_energy_matrix_rows(row_indices, kernel, domain_dim, param):
    support = _kernel_support(kernel)
    dim_x, dim_y, dim_z = domain_dim
    rows = []
    cols = []
    data = []
    for row_idx, flat_target in enumerate(row_indices):
        x_t, y_t, z_t = _flat_to_xyz(int(flat_target), domain_dim)
        for dx, dy, dz, weight in support:
            x_i = x_t + dx
            y_i = y_t + dy
            z_i = z_t + dz
            if not (0 <= x_i < dim_x and 0 <= y_i < dim_y and 0 <= z_i < dim_z):
                continue
            rows.append(row_idx)
            cols.append(_xyz_to_flat(x_i, y_i, z_i, domain_dim))
            data.append(param.exposure_time * weight)
    return sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(row_indices), dim_x * dim_y * dim_z),
        dtype=np.float64,
    )


def build_energy_matrix(kernel, domain_dim, param):
    n_voxels = int(np.prod(domain_dim))
    return _build_energy_matrix_rows(
        np.arange(n_voxels, dtype=int), kernel, domain_dim, param
    )


def build_inside_energy_matrix(C, kernel, domain_dim, param):
    target_indices = np.flatnonzero(np.asarray(C) == 1)
    return (
        _build_energy_matrix_rows(target_indices, kernel, domain_dim, param),
        target_indices,
    )


def _identity_initial_point(initial_intensity):
    return np.asarray(initial_intensity, dtype=np.float64)


def _build_guven_linear(
    problem,
    A,
    C,
    T,
    param,
    problem_parameters,
    *,
    domain_dim,
):
    del T, problem_parameters
    C_np = np.asarray(C)
    inside = np.flatnonzero(C_np == 1)
    boundary = np.flatnonzero(
        np.asarray(guven_boundary_mask(C, domain_dim), dtype=bool)
    )
    A_inside = A[inside]
    c = np.asarray(A[boundary].sum(axis=0)).ravel()
    return LinearFormulation(
        c=c,
        A_ub=-A_inside,
        b_ub=np.full(A_inside.shape[0], -float(param.E_crit)),
        A_eq=None,
        b_eq=None,
        bounds=problem.bounds(A.shape[1], param),
        intensity_size=A.shape[1],
        initial_point=_identity_initial_point,
    )


def _build_wang_linear(
    problem,
    A,
    C,
    T,
    param,
    problem_parameters,
    *,
    domain_dim,
):
    del domain_dim, problem_parameters
    C_np = np.asarray(C)
    T_np = np.asarray(T, dtype=np.float64)
    inside = np.flatnonzero(C_np == 1)
    outside = np.flatnonzero(C_np == 0)
    A_inside = A[inside]
    A_outside = A[outside]
    n = A.shape[1]
    identity = sparse.eye(n, format="csr")
    zero_inside = sparse.csr_matrix((inside.size, n))
    zero_outside = sparse.csr_matrix((outside.size, n))
    A_ub = sparse.vstack(
        (
            sparse.hstack((A, -identity)),
            sparse.hstack((-A, -identity)),
            sparse.hstack((-A_inside, zero_inside)),
            sparse.hstack((A_outside, zero_outside)),
        ),
        format="csr",
    )
    b_ub = np.concatenate(
        (
            T_np,
            -T_np,
            np.full(inside.size, -float(param.E_crit)),
            np.full(outside.size, float(param.E_crit)),
        )
    )

    def initial_point(initial_intensity):
        intensity = np.asarray(initial_intensity, dtype=np.float64)
        energy = np.asarray(A @ intensity, dtype=np.float64)
        return np.concatenate((intensity, np.abs(energy - T_np)))

    return LinearFormulation(
        c=np.concatenate((np.zeros(n), np.ones(n))),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=None,
        b_eq=None,
        bounds=problem.bounds(n, param) + [(0.0, None) for _ in range(n)],
        intensity_size=n,
        initial_point=initial_point,
    )


def _build_reverse_guven_linear(
    problem,
    A,
    C,
    T,
    param,
    problem_parameters,
    *,
    domain_dim,
):
    del T, domain_dim, problem_parameters
    C_np = np.asarray(C)
    inside = np.flatnonzero(C_np == 1)
    outside = np.flatnonzero(C_np == 0)
    c = -np.asarray(A[inside].sum(axis=0)).ravel()
    A_outside = A[outside]
    return LinearFormulation(
        c=c,
        A_ub=A_outside,
        b_ub=np.full(outside.size, float(param.E_crit)),
        A_eq=None,
        b_eq=None,
        bounds=problem.bounds(A.shape[1], param),
        intensity_size=A.shape[1],
        initial_point=_identity_initial_point,
    )


def _build_cure_linear(
    problem,
    A,
    C,
    T,
    param,
    problem_parameters,
    *,
    domain_dim,
):
    del T, domain_dim, problem_parameters
    C_np = np.asarray(C)
    inside = np.flatnonzero(C_np == 1)
    outside = np.flatnonzero(C_np == 0)
    A_inside = A[inside]
    A_outside = A[outside]
    n = A.shape[1]
    zero_inside_outside = sparse.csr_matrix((inside.size, outside.size))
    zero_outside_inside = sparse.csr_matrix((outside.size, inside.size))
    A_ub = sparse.vstack(
        (
            sparse.hstack(
                (-A_inside, -sparse.eye(inside.size), zero_inside_outside)
            ),
            sparse.hstack(
                (A_outside, zero_outside_inside, -sparse.eye(outside.size))
            ),
        ),
        format="csr",
    )
    b_ub = np.concatenate(
        (
            np.full(inside.size, -float(param.E_crit)),
            np.full(outside.size, float(param.E_crit)),
        )
    )

    def initial_point(initial_intensity):
        intensity = np.asarray(initial_intensity, dtype=np.float64)
        energy = np.asarray(A @ intensity, dtype=np.float64)
        return np.concatenate(
            (
                intensity,
                np.maximum(0.0, float(param.E_crit) - energy[inside]),
                np.maximum(0.0, energy[outside] - float(param.E_crit)),
            )
        )

    return LinearFormulation(
        c=np.concatenate((np.zeros(n), np.ones(inside.size + outside.size))),
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=None,
        b_eq=None,
        bounds=(
            problem.bounds(n, param)
            + [(0.0, None) for _ in range(inside.size + outside.size)]
        ),
        intensity_size=n,
        initial_point=initial_point,
    )


LINEAR_BUILDERS = {
    "obj_fun_2": _build_cure_linear,
    "guven": _build_guven_linear,
    "wang": _build_wang_linear,
    "reverse_guven": _build_reverse_guven_linear,
}


def _build_linear(
    problem,
    *,
    T,
    C,
    kernel,
    domain_dim,
    param,
    problem_parameters,
    penalty_parameters,
    **_kwargs,
):
    del penalty_parameters
    A = build_energy_matrix(kernel, domain_dim, param)
    return LINEAR_BUILDERS[problem.name](
        problem,
        A,
        C,
        T,
        param,
        problem_parameters,
        domain_dim=domain_dim,
    )


FORMULATION_ADAPTERS: dict[str, FormulationAdapter] = {
    "penalty": FormulationAdapter(
        name="penalty",
        result_label="exterior_penalty",
        constraint_handling="penalty",
        supports=lambda problem: problem.name in PENALTY_POLICIES,
        is_smooth=lambda problem: PENALTY_POLICIES[problem.name].smooth,
        build=_build_penalty_objective,
    ),
    "constrained": FormulationAdapter(
        name="constrained",
        result_label="direct_constraints",
        constraint_handling="direct",
        supports=lambda problem: problem.has_constraints,
        is_smooth=lambda problem: problem.objective_is_smooth,
        build=_build_constrained,
    ),
    "linear": FormulationAdapter(
        name="linear",
        result_label="linear_program",
        constraint_handling=(
            lambda problem: "direct" if problem.has_constraints else "none"
        ),
        supports=lambda problem: problem.name in LINEAR_BUILDERS,
        is_smooth=lambda problem: True,
        build=_build_linear,
    ),
    "objective": FormulationAdapter(
        name="objective",
        result_label="base_objective",
        constraint_handling="none",
        supports=lambda problem: not problem.has_constraints,
        is_smooth=lambda problem: problem.objective_is_smooth,
        build=_build_objective,
    ),
}
