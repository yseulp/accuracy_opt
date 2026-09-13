"""Solver implementations and formulation-based compatibility dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy import sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint, linprog, minimize

from scripts.optimization.formulations import (
    FORMULATION_ADAPTERS,
    FunctionalConstraintSpec,
    LinearConstraintSpec,
    resolve_penalty_parameters,
)
from scripts.optimization.problems import PROBLEMS


@dataclass(frozen=True)
class OptimizationResult:
    I_opt: jnp.ndarray
    success: bool
    converged: bool | None
    message: str
    iterations: int | None
    initial_objective: float | None
    final_objective: float | None
    final_gradient_norm: float | None
    runtime_seconds: float
    formulation: str


@dataclass(frozen=True)
class FormulationHandler:
    solve: Callable
    accepts: Callable[[Any], bool] | None = None
    incompatibility_reason: str | None = None


@dataclass(frozen=True)
class SolverDefinition:
    formulation_handlers: dict[str, FormulationHandler]
    parameter_defaults: dict[str, Any]


def solve_pgd(formulation, *, I_start, solver_parameters):
    """Projected gradient descent with box projection."""
    step_size = solver_parameters["step_size"]
    max_iterations = solver_parameters["max_iterations"]
    objective_tolerance = solver_parameters["objective_tolerance"]
    lower_bounds = jnp.asarray([bound[0] for bound in formulation.bounds])
    upper_bounds = jnp.asarray([bound[1] for bound in formulation.bounds])
    I = jnp.clip(jnp.asarray(I_start), lower_bounds, upper_bounds)
    value_and_grad = jax.value_and_grad(formulation.objective)
    start_time = time.perf_counter()

    objective_array, gradient = value_and_grad(I)
    objective_array, gradient = jax.block_until_ready((objective_array, gradient))
    current_objective = float(objective_array)
    current_gradient_norm = float(jnp.linalg.norm(gradient))
    initial_objective = (
        current_objective if np.isfinite(current_objective) else None
    )
    iterations = 0
    success = True
    converged = None if objective_tolerance is None else False
    message = "maximum iterations completed"

    for iteration in range(1, max_iterations + 1):
        if not np.isfinite(current_objective) or not np.all(
            np.isfinite(np.asarray(gradient))
        ):
            success = False
            message = "objective or gradient is not finite"
            break

        candidate = jnp.clip(
            I - step_size * gradient,
            lower_bounds,
            upper_bounds,
        )
        next_objective_array, next_gradient = value_and_grad(candidate)
        candidate, next_objective_array, next_gradient = jax.block_until_ready(
            (candidate, next_objective_array, next_gradient)
        )
        next_objective = float(next_objective_array)

        if not np.isfinite(next_objective) or not np.all(
            np.isfinite(np.asarray(next_gradient))
        ):
            success = False
            message = "updated objective or gradient is not finite"
            break

        objective_change = abs(current_objective - next_objective)
        I = candidate
        current_objective = next_objective
        gradient = next_gradient
        current_gradient_norm = float(jnp.linalg.norm(next_gradient))
        iterations = iteration

        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            print(
                f"  iteration {iteration:4d}/{max_iterations}: "
                f"objective={current_objective:.8g}, "
                f"|grad|={current_gradient_norm:.8g}"
            )

        if (
            objective_tolerance is not None
            and objective_change <= objective_tolerance
        ):
            converged = True
            message = (
                f"objective tolerance reached: |delta| <= {objective_tolerance}"
            )
            break

    if success and objective_tolerance is not None and not converged:
        success = False
        message = "maximum iterations reached before objective tolerance"

    return OptimizationResult(
        I_opt=I,
        success=success,
        converged=converged,
        message=message,
        iterations=iterations,
        initial_objective=initial_objective,
        final_objective=(current_objective if np.isfinite(current_objective) else None),
        final_gradient_norm=(
            current_gradient_norm if np.isfinite(current_gradient_norm) else None
        ),
        runtime_seconds=time.perf_counter() - start_time,
        formulation="objective",
    )


def solve_slsqp_objective(formulation, *, I_start, solver_parameters):
    """SLSQP for a smooth scalar objective with intensity bounds."""
    value_and_grad = jax.value_and_grad(formulation.objective)

    def objective(I):
        value, _ = value_and_grad(jnp.asarray(I))
        return float(value)

    def objective_grad(I):
        _, gradient = value_and_grad(jnp.asarray(I))
        return np.asarray(gradient, dtype=np.float64)

    initial_objective = objective(I_start)
    start_time = time.perf_counter()
    result = minimize(
        objective,
        x0=np.asarray(I_start, dtype=np.float64),
        jac=objective_grad,
        method="SLSQP",
        bounds=formulation.bounds,
        options={
            "ftol": solver_parameters["ftol"],
            "maxiter": solver_parameters["max_iterations"],
            "disp": True,
        },
    )
    runtime_seconds = time.perf_counter() - start_time
    final_gradient = objective_grad(result.x)

    return OptimizationResult(
        I_opt=jnp.asarray(result.x),
        success=bool(result.success),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=initial_objective,
        final_objective=objective(result.x),
        final_gradient_norm=float(np.linalg.norm(final_gradient)),
        runtime_seconds=runtime_seconds,
        formulation="objective",
    )


def _build_scipy_constraints(formulation):
    constraints = []
    for constraint in formulation.constraints:
        if isinstance(constraint, FunctionalConstraintSpec):
            constraints.append(
                NonlinearConstraint(
                    constraint.function,
                    constraint.lower_bound,
                    constraint.upper_bound,
                    jac=constraint.jacobian,
                )
            )
        elif isinstance(constraint, LinearConstraintSpec):
            constraints.append(
                LinearConstraint(
                    constraint.matrix,
                    constraint.lower_bound,
                    constraint.upper_bound,
                )
            )
        else:
            raise TypeError(
                f"Unsupported constraint specification: {type(constraint).__name__}"
            )
    return constraints


def solve_slsqp_constrained(formulation, *, I_start, solver_parameters):
    """SLSQP for an objective with explicit linear or nonlinear constraints."""
    constraints = _build_scipy_constraints(formulation)
    initial_objective = float(formulation.objective(I_start))
    start_time = time.perf_counter()
    result = minimize(
        formulation.objective,
        x0=np.asarray(I_start, dtype=np.float64),
        jac=formulation.objective_grad,
        method="SLSQP",
        bounds=formulation.bounds,
        constraints=constraints,
        options={
            "ftol": solver_parameters["ftol"],
            "maxiter": solver_parameters["max_iterations"],
            "disp": True,
        },
    )
    runtime_seconds = time.perf_counter() - start_time
    final_gradient = formulation.objective_grad(result.x)

    return OptimizationResult(
        I_opt=jnp.asarray(result.x),
        success=bool(result.success),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=initial_objective,
        final_objective=float(formulation.objective(result.x)),
        final_gradient_norm=float(np.linalg.norm(final_gradient)),
        runtime_seconds=runtime_seconds,
        formulation="explicit_constraints",
    )


def solve_trust_constr_objective(formulation, *, I_start, solver_parameters):
    """trust-constr for a smooth scalar objective with intensity bounds."""
    value_and_grad = jax.value_and_grad(formulation.objective)

    def objective(I):
        value, _ = value_and_grad(jnp.asarray(I))
        return float(value)

    def objective_grad(I):
        _, gradient = value_and_grad(jnp.asarray(I))
        return np.asarray(gradient, dtype=np.float64)

    start_time = time.perf_counter()
    result = minimize(
        objective,
        x0=np.asarray(I_start, dtype=np.float64),
        jac=objective_grad,
        method="trust-constr",
        bounds=formulation.bounds,
        options={
            "maxiter": solver_parameters["max_iterations"],
            "gtol": solver_parameters["gtol"],
            "xtol": solver_parameters["xtol"],
            "barrier_tol": solver_parameters["barrier_tol"],
            "verbose": 1,
        },
    )
    return OptimizationResult(
        I_opt=jnp.asarray(result.x),
        success=bool(result.success),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=objective(I_start),
        final_objective=objective(result.x),
        final_gradient_norm=float(np.linalg.norm(objective_grad(result.x))),
        runtime_seconds=time.perf_counter() - start_time,
        formulation="objective",
    )


def solve_trust_constr_constrained(
    formulation,
    *,
    I_start,
    solver_parameters,
):
    """trust-constr for a smooth objective with explicit constraints."""
    start_time = time.perf_counter()
    result = minimize(
        formulation.objective,
        x0=np.asarray(I_start, dtype=np.float64),
        jac=formulation.objective_grad,
        method="trust-constr",
        bounds=formulation.bounds,
        constraints=_build_scipy_constraints(formulation),
        options={
            "maxiter": solver_parameters["max_iterations"],
            "gtol": solver_parameters["gtol"],
            "xtol": solver_parameters["xtol"],
            "barrier_tol": solver_parameters["barrier_tol"],
            "verbose": 1,
        },
    )
    return OptimizationResult(
        I_opt=jnp.asarray(result.x),
        success=bool(result.success),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=float(formulation.objective(I_start)),
        final_objective=float(formulation.objective(result.x)),
        final_gradient_norm=float(
            np.linalg.norm(formulation.objective_grad(result.x))
        ),
        runtime_seconds=time.perf_counter() - start_time,
        formulation="direct_constraints",
    )


def solve_trust_constr_linear(formulation, *, I_start, solver_parameters):
    """trust-constr using only the supplied complete linear formulation."""
    initial_point = formulation.initial_point(
        np.asarray(I_start, dtype=np.float64)
    )

    def objective(x):
        return float(np.dot(formulation.c, x))

    def objective_grad(_x):
        return formulation.c

    objective_hessian = sparse.csr_matrix(
        (len(formulation.c), len(formulation.c)), dtype=np.float64
    )

    def objective_hess(_x):
        return objective_hessian

    constraints = []
    if formulation.A_ub is not None:
        constraints.append(
            LinearConstraint(formulation.A_ub, lb=-np.inf, ub=formulation.b_ub)
        )
    if formulation.A_eq is not None:
        constraints.append(
            LinearConstraint(
                formulation.A_eq,
                lb=formulation.b_eq,
                ub=formulation.b_eq,
            )
        )

    start_time = time.perf_counter()
    result = minimize(
        objective,
        x0=initial_point,
        jac=objective_grad,
        hess=objective_hess,
        method="trust-constr",
        bounds=formulation.bounds,
        constraints=constraints,
        options={
            "maxiter": solver_parameters["max_iterations"],
            "gtol": solver_parameters["gtol"],
            "xtol": solver_parameters["xtol"],
            "barrier_tol": solver_parameters["barrier_tol"],
            "verbose": 1,
        },
    )
    runtime_seconds = time.perf_counter() - start_time

    return OptimizationResult(
        I_opt=jnp.asarray(formulation.extract_intensity(result.x)),
        success=bool(result.success),
        converged=bool(result.success),
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=objective(initial_point),
        final_objective=objective(result.x),
        final_gradient_norm=None,
        runtime_seconds=runtime_seconds,
        formulation="linear_constraints",
    )


def solve_linprog(formulation, *, I_start, solver_parameters):
    """HiGHS using only the supplied complete linear formulation."""
    del solver_parameters
    initial_point = formulation.initial_point(
        np.asarray(I_start, dtype=np.float64)
    )
    initial_objective = float(np.dot(formulation.c, initial_point))
    start_time = time.perf_counter()
    result = linprog(
        c=formulation.c,
        A_ub=formulation.A_ub,
        b_ub=formulation.b_ub,
        A_eq=formulation.A_eq,
        b_eq=formulation.b_eq,
        bounds=formulation.bounds,
        method="highs",
    )
    runtime_seconds = time.perf_counter() - start_time

    if not result.success or result.x is None:
        return OptimizationResult(
            I_opt=jnp.asarray(I_start),
            success=False,
            converged=False,
            message=f"HiGHS optimization failed: {result.message}",
            iterations=int(result.nit) if result.nit is not None else None,
            initial_objective=initial_objective,
            final_objective=initial_objective,
            final_gradient_norm=None,
            runtime_seconds=runtime_seconds,
            formulation="linear_program",
        )

    return OptimizationResult(
        I_opt=jnp.asarray(formulation.extract_intensity(result.x)),
        success=True,
        converged=True,
        message=str(result.message),
        iterations=int(result.nit) if result.nit is not None else None,
        initial_objective=initial_objective,
        final_objective=float(result.fun),
        final_gradient_norm=None,
        runtime_seconds=runtime_seconds,
        formulation="linear_program",
    )


def _accepts_smooth_formulation(descriptor) -> bool:
    return bool(descriptor.smooth)


SOLVERS: dict[str, SolverDefinition] = {
    "projected_gradient_descent": SolverDefinition(
        formulation_handlers={
            "penalty": FormulationHandler(solve=solve_pgd),
            "objective": FormulationHandler(
                solve=solve_pgd,
            ),
        },
        parameter_defaults={
            "step_size": 0.03,
            "max_iterations": 100,
            "objective_tolerance": None,
        },
    ),
    "slsqp": SolverDefinition(
        formulation_handlers={
            "constrained": FormulationHandler(
                solve=solve_slsqp_constrained,
                accepts=_accepts_smooth_formulation,
                incompatibility_reason="SLSQP requires a smooth formulation",
            ),
            "objective": FormulationHandler(
                solve=solve_slsqp_objective,
                accepts=_accepts_smooth_formulation,
                incompatibility_reason=(
                    "SLSQP requires a smooth objective"
                ),
            ),
        },
        parameter_defaults={"max_iterations": 100, "ftol": 1e-6},
    ),
    "trust_constr": SolverDefinition(
        formulation_handlers={
            "linear": FormulationHandler(
                solve=solve_trust_constr_linear,
            ),
            "constrained": FormulationHandler(
                solve=solve_trust_constr_constrained,
                accepts=_accepts_smooth_formulation,
                incompatibility_reason="trust-constr requires a smooth formulation",
            ),
            "objective": FormulationHandler(
                solve=solve_trust_constr_objective,
                accepts=_accepts_smooth_formulation,
                incompatibility_reason="trust-constr requires a smooth objective",
            ),
        },
        parameter_defaults={
            "max_iterations": 250,
            "gtol": 1e-4,
            "xtol": 1e-4,
            "barrier_tol": 1e-4,
        },
    ),
    "linprog_highs": SolverDefinition(
        formulation_handlers={
            "linear": FormulationHandler(
                solve=solve_linprog,
            ),
        },
        parameter_defaults={},
    ),
}

SOLVER_ALIASES = {
    "highs": "linprog_highs",
    "linprog": "linprog_highs",
    "trust-constr": "trust_constr",
}


def normalize_solver_name(solver_name: str) -> str:
    return SOLVER_ALIASES.get(solver_name, solver_name)


def check_compatibility(problem_name: str, solver_name: str) -> str:
    """Return the highest-priority compatible formulation for a combination."""
    canonical_solver = normalize_solver_name(solver_name)
    if problem_name not in PROBLEMS:
        raise ValueError(
            f"Unknown problem {problem_name!r}. Available: {', '.join(PROBLEMS)}"
        )
    if canonical_solver not in SOLVERS:
        available = sorted(set(SOLVERS) | set(SOLVER_ALIASES))
        raise ValueError(
            f"Unknown solver {solver_name!r}. Available: {', '.join(available)}"
        )

    problem = PROBLEMS[problem_name]
    solver = SOLVERS[canonical_solver]
    rejected_reasons = []
    for adapter_name, handler in solver.formulation_handlers.items():
        adapter = FORMULATION_ADAPTERS[adapter_name]
        if not adapter.supports(problem):
            continue
        descriptor = adapter.describe(problem)
        if handler.accepts is None or handler.accepts(descriptor):
            return adapter_name
        if handler.incompatibility_reason:
            rejected_reasons.append(handler.incompatibility_reason)

    accepted = ", ".join(solver.formulation_handlers)
    provided = ", ".join(
        name
        for name, adapter in FORMULATION_ADAPTERS.items()
        if adapter.supports(problem)
    )
    detail = f" {'; '.join(rejected_reasons)}." if rejected_reasons else ""
    raise ValueError(
        f"Incompatible combination: problem {problem_name!r} + solver "
        f"{canonical_solver!r}. The solver accepts formulations: {accepted}. "
        f"The problem provides: {provided}.{detail}"
    )


def resolve_selection(
    problem_name: str,
    solver_name: str,
    *,
    problem_defaults: dict[str, Any] | None = None,
    penalty_defaults: dict[str, Any] | None = None,
    solver_defaults: dict[str, Any] | None = None,
    problem_parameters: dict[str, Any] | None = None,
    penalty_parameters: dict[str, Any] | None = None,
    solver_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve one independent problem and solver selection."""
    if not isinstance(problem_name, str) or not problem_name:
        raise ValueError("problem must be a non-empty string.")
    if not isinstance(solver_name, str) or not solver_name:
        raise ValueError("solver must be a non-empty string.")

    canonical_solver = normalize_solver_name(solver_name)
    adapter_name = check_compatibility(problem_name, canonical_solver)
    problem = PROBLEMS[problem_name]
    configured_problem_defaults = _mapping(
        {} if problem_defaults is None else problem_defaults,
        "configured problem defaults",
    )
    problem_overrides = _mapping(
        {} if problem_parameters is None else problem_parameters,
        "problem_parameters",
    )
    configured_penalty_defaults = _mapping(
        {} if penalty_defaults is None else penalty_defaults,
        "configured penalty defaults",
    )
    penalty_overrides = _mapping(
        {} if penalty_parameters is None else penalty_parameters,
        "penalty_parameters",
    )
    if adapter_name != "penalty" and penalty_overrides:
        raise ValueError(
            f"Problem {problem_name!r} with solver {canonical_solver!r} uses "
            "direct constraint handling and does not accept penalty parameters."
        )
    configured_solver_defaults = _mapping(
        {} if solver_defaults is None else solver_defaults,
        "configured solver defaults",
    )
    solver_overrides = _mapping(
        {} if solver_parameters is None else solver_parameters,
        "solver_parameters",
    )

    descriptor = FORMULATION_ADAPTERS[adapter_name].describe(problem)
    return {
        "problem": problem_name,
        "solver": canonical_solver,
        "problem_parameters": problem.resolve_parameters(
            {**configured_problem_defaults, **problem_overrides},
        ),
        "penalty_parameters": (
            resolve_penalty_parameters(
                problem_name,
                {**configured_penalty_defaults, **penalty_overrides},
            )
            if adapter_name == "penalty"
            else {}
        ),
        "solver_parameters": resolve_solver_parameters(
            canonical_solver,
            {**configured_solver_defaults, **solver_overrides},
        ),
        "adapter": adapter_name,
        "formulation": descriptor.result_label,
        "constraint_handling": descriptor.constraint_handling,
    }


def run_solver(
    solver_name: str,
    problem_name: str,
    *,
    I_start,
    T,
    C,
    kernel,
    domain_dim,
    param,
    problem_parameters: dict[str, Any],
    penalty_parameters: dict[str, Any] | None = None,
    solver_parameters: dict[str, Any],
    adapter_name: str | None = None,
) -> OptimizationResult:
    """Build the required problem representation and pass it to the solver."""
    canonical_solver = normalize_solver_name(solver_name)
    selected_adapter_name = check_compatibility(problem_name, canonical_solver)
    if adapter_name is not None and adapter_name != selected_adapter_name:
        raise ValueError(
            f"Adapter {adapter_name!r} does not match the resolved adapter "
            f"{selected_adapter_name!r}."
        )
    adapter_name = selected_adapter_name
    problem = PROBLEMS[problem_name]
    solver = SOLVERS[canonical_solver]
    adapter = FORMULATION_ADAPTERS[adapter_name]
    handler = solver.formulation_handlers[adapter_name]
    resolved_problem_parameters = problem.resolve_parameters(problem_parameters)
    if adapter_name != "penalty" and penalty_parameters:
        raise ValueError(
            f"Adapter {adapter_name!r} uses direct or no constraint handling "
            "and does not accept penalty parameters."
        )
    resolved_penalty_parameters = (
        resolve_penalty_parameters(problem_name, penalty_parameters or {})
        if adapter_name == "penalty"
        else {}
    )
    resolved_solver_parameters = resolve_solver_parameters(
        canonical_solver, solver_parameters
    )
    total_start_time = time.perf_counter()
    formulation = adapter.build(
        problem,
        I_start=I_start,
        T=T,
        C=C,
        kernel=kernel,
        domain_dim=domain_dim,
        param=param,
        problem_parameters=resolved_problem_parameters,
        penalty_parameters=resolved_penalty_parameters,
    )
    result = handler.solve(
        formulation,
        I_start=I_start,
        solver_parameters=resolved_solver_parameters,
    )
    return replace(
        result,
        runtime_seconds=time.perf_counter() - total_start_time,
        formulation=adapter.result_label,
    )


def resolve_solver_parameters(
    solver_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    defaults = SOLVERS[solver_name].parameter_defaults
    unknown = set(parameters) - set(defaults)
    if unknown:
        raise ValueError(
            f"Solver {solver_name!r} does not use parameter(s): "
            + ", ".join(sorted(unknown))
        )
    resolved = {**defaults, **parameters}

    if "max_iterations" in resolved:
        resolved["max_iterations"] = _positive_int(
            resolved["max_iterations"], f"{solver_name}.max_iterations"
        )
    for name in ("step_size", "ftol", "gtol", "xtol", "barrier_tol"):
        if name in resolved:
            resolved[name] = _positive_float(
                resolved[name], f"{solver_name}.{name}"
            )
    if "objective_tolerance" in resolved:
        value = resolved["objective_tolerance"]
        resolved["objective_tolerance"] = (
            None
            if value is None
            else _nonnegative_float(value, f"{solver_name}.objective_tolerance")
        )
    return resolved


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _positive_float(value: Any, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return resolved


def _nonnegative_float(value: Any, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative.")
    return resolved


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be a finite number.")
    return resolved
