"""Standalone per-iteration history analysis for configured optimization runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import yaml
from scipy import sparse
from scipy.optimize import LinearConstraint, NonlinearConstraint, linprog, minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.benchmark.config import load_benchmark_config
from scripts.benchmark.core import BenchmarkData, build_benchmark
from scripts.model.simulation import simulate
from scripts.optimization.config import (
    DEFAULT_OPTIMIZATION_CONFIG_PATH,
    defaults_for_selection,
    load_optimization_defaults,
)
from scripts.optimization.formulations import (
    FORMULATION_ADAPTERS,
    PENALTY_POLICIES,
    ConstrainedFormulation,
    FunctionalConstraintSpec,
    LinearConstraintSpec,
    LinearFormulation,
    ObjectiveFormulation,
)
from scripts.optimization.problems import PROBLEMS
from scripts.optimization.solvers import (
    SOLVERS,
    SOLVER_ALIASES,
    normalize_solver_name,
    resolve_selection,
)


@dataclass(frozen=True)
class HistoryRecord:
    iteration: int | None
    record_type: str
    original_objective: float
    raw_constraint_violation: float | None
    raw_penalty: float | None
    weighted_penalty: float | None
    formulation_objective: float
    min_constraint_margin: float | None
    violated_constraint_count: int | None
    mean_constraint_violation: float | None
    max_constraint_violation: float | None
    min_inside_constraint_margin: float | None
    min_outside_constraint_margin: float | None
    undercured_voxels: int
    overcured_voxels: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record and plot the optimization history of one benchmark run."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--optimization-config",
        type=Path,
        default=DEFAULT_OPTIMIZATION_CONFIG_PATH,
    )
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument(
        "--solver",
        choices=sorted(set(SOLVERS) | set(SOLVER_ALIASES)),
        required=True,
    )
    parser.add_argument("--problem-param", action="append", default=None)
    parser.add_argument("--penalty-param", action="append", default=None)
    parser.add_argument("--solver-param", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact history output directory; defaults to "
            "results/optimization_history/<benchmark>/<problem>/<solver>."
        ),
    )
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF plot.")
    return parser.parse_args()


def run_history_analysis(
    *,
    benchmark_config_path: Path,
    optimization_config_path: Path,
    problem_name: str,
    solver_name: str,
    problem_parameters: dict[str, Any] | None = None,
    penalty_parameters: dict[str, Any] | None = None,
    solver_parameters: dict[str, Any] | None = None,
    output_directory: Path | None = None,
    save_pdf: bool = False,
) -> tuple[list[HistoryRecord], dict[str, Any]]:
    benchmark_config_path = benchmark_config_path.resolve()
    optimization_config_path = optimization_config_path.resolve()
    benchmark_config = load_benchmark_config(benchmark_config_path)
    optimization_config = load_optimization_defaults(optimization_config_path)
    problem_defaults, penalty_defaults, solver_defaults = defaults_for_selection(
        optimization_config, problem_name, solver_name
    )
    selection = resolve_selection(
        problem_name,
        solver_name,
        problem_defaults=problem_defaults,
        penalty_defaults=penalty_defaults,
        solver_defaults=solver_defaults,
        problem_parameters=problem_parameters or {},
        penalty_parameters=penalty_parameters or {},
        solver_parameters=solver_parameters or {},
    )
    benchmark = build_benchmark(benchmark_config)
    problem = PROBLEMS[problem_name]
    adapter = FORMULATION_ADAPTERS[selection["adapter"]]
    formulation = adapter.build(
        problem,
        I_start=benchmark.I_start,
        T=benchmark.T,
        C=benchmark.C,
        kernel=benchmark.kernel,
        domain_dim=benchmark.domain_dim,
        param=benchmark.param,
        problem_parameters=selection["problem_parameters"],
        penalty_parameters=selection["penalty_parameters"],
    )

    if output_directory is None:
        output_directory = _default_output_directory(
            benchmark_config,
            problem_name,
            selection["solver"],
        )
    else:
        output_directory = output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    runner = HISTORY_RUNNERS[selection["solver"]]
    history, solver_metadata = runner(
        formulation=formulation,
        benchmark=benchmark,
        problem=problem,
        selection=selection,
    )
    metadata = {
        "benchmark": benchmark_config["benchmark"]["name"],
        "benchmark_config": str(benchmark_config_path),
        "optimization_config": str(optimization_config_path),
        "problem": problem_name,
        "solver": selection["solver"],
        "output_directory": str(output_directory),
        "adapter": selection["adapter"],
        "formulation": selection["formulation"],
        "constraint_handling": selection["constraint_handling"],
        "problem_parameters": selection["problem_parameters"],
        "penalty_parameters": selection["penalty_parameters"],
        "solver_parameters": selection["solver_parameters"],
        **solver_metadata,
    }
    _write_history_csv(history, output_directory / "optimization_history.csv")
    _write_json(metadata, output_directory / "optimization_history_metadata.json")
    create_history_plot(
        history,
        metadata,
        output_directory / "optimization_history.png",
    )
    if save_pdf:
        create_history_plot(
            history,
            metadata,
            output_directory / "optimization_history.pdf",
        )
    print(f"Optimization history saved in: {output_directory}")
    return history, metadata


def _run_pgd_history(*, formulation, benchmark, problem, selection):
    parameters = selection["solver_parameters"]
    lower = jnp.asarray([bound[0] for bound in formulation.bounds])
    upper = jnp.asarray([bound[1] for bound in formulation.bounds])
    intensity = jnp.clip(benchmark.I_start, lower, upper)
    value_and_grad = jax.value_and_grad(formulation.objective)
    history = []
    start_time = time.perf_counter()

    objective, gradient = value_and_grad(intensity)
    objective, gradient = jax.block_until_ready((objective, gradient))
    history.append(
        _evaluate_state(
            0,
            "initial",
            intensity,
            float(objective),
            benchmark,
            problem,
            selection,
        )
    )
    success = _finite_objective_and_gradient(objective, gradient)
    objective_tolerance = parameters["objective_tolerance"]
    converged = None if objective_tolerance is None else False
    message = (
        "maximum iterations completed"
        if success
        else "objective or gradient is not finite"
    )
    for iteration in range(1, parameters["max_iterations"] + 1):
        if not success:
            break
        candidate = jnp.clip(
            intensity - parameters["step_size"] * gradient,
            lower,
            upper,
        )
        next_objective, next_gradient = value_and_grad(candidate)
        candidate, next_objective, next_gradient = jax.block_until_ready(
            (candidate, next_objective, next_gradient)
        )
        if not _finite_objective_and_gradient(next_objective, next_gradient):
            success = False
            message = "updated objective or gradient is not finite"
            break
        objective_change = abs(float(objective) - float(next_objective))
        intensity = candidate
        objective = next_objective
        gradient = next_gradient
        history.append(
            _evaluate_state(
                iteration,
                "iteration",
                intensity,
                float(objective),
                benchmark,
                problem,
                selection,
            )
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

    return history, {
        "history_available": True,
        "iteration_semantics": "one projected gradient update",
        "reported_iterations": history[-1].iteration,
        "success": success,
        "converged": converged,
        "message": message,
        "runtime_seconds": time.perf_counter() - start_time,
    }


def _run_slsqp_history(*, formulation, benchmark, problem, selection):
    parameters = selection["solver_parameters"]
    objective, objective_grad, bounds, constraints, initial_point, extract = (
        _scipy_formulation_parts(formulation, benchmark.I_start)
    )
    history = [
        _evaluate_state(
            0,
            "initial",
            extract(initial_point),
            objective(initial_point),
            benchmark,
            problem,
            selection,
        )
    ]
    callback_iteration = 0
    last_x = np.asarray(initial_point)

    def callback(xk):
        nonlocal callback_iteration, last_x
        callback_iteration += 1
        last_x = np.asarray(xk).copy()
        history.append(
            _evaluate_state(
                callback_iteration,
                "iteration",
                extract(xk),
                objective(xk),
                benchmark,
                problem,
                selection,
            )
        )

    start_time = time.perf_counter()
    result = minimize(
        objective,
        x0=initial_point,
        jac=objective_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={
            "ftol": parameters["ftol"],
            "maxiter": parameters["max_iterations"],
            "disp": True,
        },
    )
    runtime = time.perf_counter() - start_time
    _append_final_if_needed(
        history,
        last_x,
        result.x,
        int(result.nit),
        objective,
        extract,
        benchmark,
        problem,
        selection,
    )
    return history, {
        "history_available": True,
        "iteration_semantics": "one SLSQP major iteration callback",
        "reported_iterations": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "runtime_seconds": runtime,
    }


def _run_trust_constr_history(*, formulation, benchmark, problem, selection):
    parameters = selection["solver_parameters"]
    objective, objective_grad, bounds, constraints, initial_point, extract = (
        _scipy_formulation_parts(formulation, benchmark.I_start)
    )
    minimize_arguments = {}
    if isinstance(formulation, LinearFormulation):
        zero_hessian = sparse.csr_matrix(
            (len(formulation.c), len(formulation.c)), dtype=np.float64
        )
        minimize_arguments["hess"] = lambda _x: zero_hessian
    history = [
        _evaluate_state(
            0,
            "initial",
            extract(initial_point),
            objective(initial_point),
            benchmark,
            problem,
            selection,
        )
    ]
    last_x = np.asarray(initial_point)

    def callback(xk, state):
        nonlocal last_x
        last_x = np.asarray(xk).copy()
        history.append(
            _evaluate_state(
                int(state.nit),
                "iteration",
                extract(xk),
                float(state.fun),
                benchmark,
                problem,
                selection,
            )
        )
        return False

    start_time = time.perf_counter()
    result = minimize(
        objective,
        x0=initial_point,
        jac=objective_grad,
        method="trust-constr",
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={
            "maxiter": parameters["max_iterations"],
            "gtol": parameters["gtol"],
            "xtol": parameters["xtol"],
            "barrier_tol": parameters["barrier_tol"],
            "verbose": 1,
        },
        **minimize_arguments,
    )
    runtime = time.perf_counter() - start_time
    _append_final_if_needed(
        history,
        last_x,
        result.x,
        int(result.nit),
        objective,
        extract,
        benchmark,
        problem,
        selection,
    )
    return history, {
        "history_available": True,
        "iteration_semantics": "trust-constr callback state.nit",
        "reported_iterations": int(result.nit),
        "success": bool(result.success),
        "message": str(result.message),
        "runtime_seconds": runtime,
    }


def _run_linprog_history(*, formulation, benchmark, problem, selection):
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
    runtime = time.perf_counter() - start_time
    solution_available = bool(result.success and result.x is not None)
    if solution_available:
        full_solution = result.x
        record_type = "final_only"
    else:
        full_solution = formulation.initial_point(benchmark.I_start)
        record_type = "failed_fallback"
    history = [
        _evaluate_state(
            None,
            record_type,
            formulation.extract_intensity(full_solution),
            float(np.dot(formulation.c, full_solution)),
            benchmark,
            problem,
            selection,
        )
    ]
    return history, {
        "history_available": False,
        "solution_available": solution_available,
        "history_limitation": (
            "scipy.optimize.linprog(method='highs') does not expose iterate "
            "callbacks; only the final state and res.nit are available"
        ),
        "iteration_semantics": None,
        "reported_iterations": (
            int(result.nit) if getattr(result, "nit", None) is not None else None
        ),
        "success": bool(result.success),
        "message": str(result.message),
        "runtime_seconds": runtime,
    }


HISTORY_RUNNERS = {
    "projected_gradient_descent": _run_pgd_history,
    "slsqp": _run_slsqp_history,
    "trust_constr": _run_trust_constr_history,
    "linprog_highs": _run_linprog_history,
}


def _scipy_formulation_parts(formulation, I_start):
    if isinstance(formulation, ObjectiveFormulation):
        value_and_grad = jax.value_and_grad(formulation.objective)

        def objective(x):
            return float(value_and_grad(jnp.asarray(x))[0])

        def objective_grad(x):
            return np.asarray(value_and_grad(jnp.asarray(x))[1], dtype=np.float64)

        return (
            objective,
            objective_grad,
            formulation.bounds,
            [],
            np.asarray(I_start, dtype=np.float64),
            lambda x: np.asarray(x, dtype=np.float64),
        )
    if isinstance(formulation, ConstrainedFormulation):
        return (
            formulation.objective,
            formulation.objective_grad,
            formulation.bounds,
            _scipy_constraints(formulation),
            np.asarray(I_start, dtype=np.float64),
            lambda x: np.asarray(x, dtype=np.float64),
        )
    if isinstance(formulation, LinearFormulation):
        objective = lambda x: float(np.dot(formulation.c, x))
        return (
            objective,
            lambda _x: formulation.c,
            formulation.bounds,
            _linear_constraints(formulation),
            formulation.initial_point(np.asarray(I_start, dtype=np.float64)),
            formulation.extract_intensity,
        )
    raise TypeError(f"Unsupported formulation: {type(formulation).__name__}")


def _scipy_constraints(formulation):
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
    return constraints


def _linear_constraints(formulation):
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
    return constraints


def _evaluate_state(
    iteration,
    record_type,
    intensity,
    formulation_objective,
    benchmark,
    problem,
    selection,
):
    intensity = jnp.asarray(intensity)
    original_objective = problem.objective(
        intensity,
        benchmark.T,
        benchmark.C,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
        selection["problem_parameters"],
    )
    residuals = problem.constraint_residuals(
        intensity,
        benchmark.T,
        benchmark.C,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
        selection["problem_parameters"],
    )
    energy = simulate(
        intensity,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
    )
    original_objective, residuals, energy = jax.block_until_ready(
        (original_objective, residuals, energy)
    )
    residuals_np = np.asarray(residuals, dtype=np.float64)
    if residuals_np.size:
        violations = np.maximum(0.0, -residuals_np)
        raw_violation = float(np.sum(violations))
        if selection["constraint_handling"] == "penalty":
            power = PENALTY_POLICIES[problem.name].power
            raw_penalty = float(np.sum(violations**power))
            weighted_penalty = (
                selection["penalty_parameters"]["penalty_weight"] * raw_penalty
            )
        else:
            raw_penalty = None
            weighted_penalty = None
        min_margin = float(np.min(residuals_np))
        violated_count = int(np.sum(residuals_np < 0.0))
        mean_violation = float(np.mean(violations))
        max_violation = float(np.max(violations))
    else:
        raw_violation = None
        raw_penalty = None
        weighted_penalty = None
        min_margin = None
        violated_count = None
        mean_violation = None
        max_violation = None

    energy_np = np.asarray(energy, dtype=np.float64)
    C_np = np.asarray(benchmark.C)
    inside_energy = energy_np[C_np == 1]
    outside_energy = energy_np[C_np == 0]
    constrained_groups = CONSTRAINT_GROUPS.get(problem.name, ())
    return HistoryRecord(
        iteration=iteration,
        record_type=record_type,
        original_objective=float(original_objective),
        raw_constraint_violation=raw_violation,
        raw_penalty=raw_penalty,
        weighted_penalty=weighted_penalty,
        formulation_objective=float(formulation_objective),
        min_constraint_margin=min_margin,
        violated_constraint_count=violated_count,
        mean_constraint_violation=mean_violation,
        max_constraint_violation=max_violation,
        min_inside_constraint_margin=(
            float(np.min(inside_energy - float(benchmark.param.E_crit)))
            if "inside" in constrained_groups
            else None
        ),
        min_outside_constraint_margin=(
            float(np.min(float(benchmark.param.E_crit) - outside_energy))
            if "outside" in constrained_groups
            else None
        ),
        undercured_voxels=int(
            np.sum(inside_energy < float(benchmark.param.E_crit))
        ),
        overcured_voxels=int(
            np.sum(outside_energy >= float(benchmark.param.E_crit))
        ),
    )


CONSTRAINT_GROUPS = {
    "guven": ("inside",),
    "wang": ("inside", "outside"),
    "reverse_guven": ("outside",),
}


def _append_final_if_needed(
    history,
    last_x,
    final_x,
    final_iteration,
    objective,
    extract,
    benchmark,
    problem,
    selection,
):
    if np.array_equal(np.asarray(last_x), np.asarray(final_x)):
        return
    final_record = _evaluate_state(
        final_iteration,
        "final",
        extract(final_x),
        objective(final_x),
        benchmark,
        problem,
        selection,
    )
    if history[-1].iteration == final_iteration:
        history[-1] = final_record
    else:
        history.append(final_record)


def create_history_plot_1(history, metadata, output_path):
    """Create compact optimization-history plots.

    Main figure:
      x-axis: solver iteration
      left y-axis: original objective and formulation/penalized objective
      right y-axis: raw constraint violation

    This makes it possible to distinguish:
      - the physical/original problem objective, and
      - the objective that the selected solver formulation actually minimizes.

    A second figure shows undercured and overcured voxel counts.
    """
    output_path = Path(output_path)

    if not metadata["history_available"]:
        figure, axis = plt.subplots(figsize=(8.0, 4.2))
        axis.axis("off")
        axis.text(
            0.5,
            0.58,
            "Iteration history unavailable",
            ha="center",
            va="center",
            fontsize=15,
            weight="bold",
        )
        axis.text(
            0.5,
            0.38,
            metadata["history_limitation"],
            ha="center",
            va="center",
            wrap=True,
        )
        figure.suptitle(f"{metadata['problem']} + {metadata['solver']}")
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        return

    iterations = np.asarray(
        [record.iteration for record in history],
        dtype=np.float64,
    )
    original_objective = np.asarray(
        [record.original_objective for record in history],
        dtype=np.float64,
    )
    formulation_objective = np.asarray(
        [record.formulation_objective for record in history],
        dtype=np.float64,
    )
    constraint_violation = np.asarray(
        [
            np.nan
            if record.raw_constraint_violation is None
            else record.raw_constraint_violation
            for record in history
        ],
        dtype=np.float64,
    )

    # Main plot:
    # - original objective J
    # - formulation objective (for penalty PGD: J + lambda * P)
    # - raw constraint violation
    figure, objective_axis = plt.subplots(figsize=(8.2, 4.8))
    constraint_axis = objective_axis.twinx()

    original_line = objective_axis.plot(
        iterations,
        original_objective,
        linewidth=2.0,
        linestyle="-",
        label="Original objective",
    )[0]

    formulation_line = objective_axis.plot(
        iterations,
        formulation_objective,
        linewidth=2.0,
        linestyle="-.",
        label="Penalized objective"
        if metadata["constraint_handling"] == "penalty"
        else "Solver objective",
    )[0]

    constraint_line = None
    if np.any(np.isfinite(constraint_violation)):
        constraint_line = constraint_axis.plot(
            iterations,
            constraint_violation,
            linewidth=2.0,
            linestyle="--",
            label="Constraint violation",
        )[0]

    objective_axis.set_xlabel("Iteration")
    objective_axis.set_ylabel("Objective")
    constraint_axis.set_ylabel("Constraint violation")
    objective_axis.set_title(
        f"Optimization history: {metadata['problem']} + {metadata['solver']}"
    )
    objective_axis.grid(True, alpha=0.25)

    lines = [original_line, formulation_line]
    if constraint_line is not None:
        lines.append(constraint_line)

    objective_axis.legend(
        lines,
        [line.get_label() for line in lines],
        loc="best",
        frameon=True,
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)

    # Second plot: physical cure quality vs. iteration.
    undercured = np.asarray(
        [record.undercured_voxels for record in history],
        dtype=np.int64,
    )
    overcured = np.asarray(
        [record.overcured_voxels for record in history],
        dtype=np.int64,
    )

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.plot(
        iterations,
        undercured,
        linewidth=2.0,
        linestyle="-",
        label="Undercured voxels",
    )
    axis.plot(
        iterations,
        overcured,
        linewidth=2.0,
        linestyle="--",
        label="Overcured voxels",
    )
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Voxel count")
    axis.set_title(
        f"Cure history: {metadata['problem']} + {metadata['solver']}"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=True)

    figure.tight_layout()
    cure_output_path = output_path.with_name(
        f"{output_path.stem}_cure{output_path.suffix}"
    )
    figure.savefig(cure_output_path, dpi=300)
    plt.close(figure)

def create_history_plot(history, metadata, output_path):
    """Create publication-style optimization-history plots."""
    output_path = Path(output_path)

    if not metadata["history_available"]:
        figure, axis = plt.subplots(figsize=(8.0, 4.2))
        axis.axis("off")
        axis.text(0.5, 0.58, "Iteration history unavailable", ha="center", va="center", fontsize=15, weight="bold")
        axis.text(0.5, 0.38, metadata["history_limitation"], ha="center", va="center", wrap=True)
        figure.suptitle(f"{metadata['problem']} + {metadata['solver']}")
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        return

    iterations = np.asarray([record.iteration for record in history], dtype=np.float64)
    original_objective = np.asarray([record.original_objective for record in history], dtype=np.float64)
    formulation_objective = np.asarray([record.formulation_objective for record in history], dtype=np.float64)
    constraint_violation = np.asarray(
        [np.nan if record.raw_constraint_violation is None else record.raw_constraint_violation for record in history],
        dtype=np.float64,
    )

    # 논문 스타일 폰트 및 설정
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "mathtext.fontset": "dejavuserif",
    })

    blue = "#1f77b4"
    red = "#d62728"

    # 1. Main Plot: Objective & Constraint Violation vs Iteration
    figure, objective_axis = plt.subplots(figsize=(6.0, 4.2))
    constraint_axis = objective_axis.twinx()

    l1 = objective_axis.plot(
        iterations, original_objective, color=blue, linestyle="-", label=r"Original Obj $J$"
    )[0]

    l2 = objective_axis.plot(
        iterations, formulation_objective, color="black", linestyle="-.", label=r"Penalized Obj $\tilde{J}$"
        if metadata["constraint_handling"] == "penalty" else "Solver Obj"
    )[0]

    lines = [l1, l2]

    if np.any(np.isfinite(constraint_violation)):
        constraint_axis.set_yscale("symlog", linthresh=1e-4)
        l3 = constraint_axis.plot(
            iterations, constraint_violation, color=red, linestyle="--", label=r"Violation $P$"
        )[0]
        lines.append(l3)

    objective_axis.set_xlabel("Iteration")
    objective_axis.set_ylabel("Objective Value", color=blue)
    constraint_axis.set_ylabel("Constraint Violation", color=red)

    objective_axis.tick_params(axis="y", colors=blue, direction="in")
    constraint_axis.tick_params(axis="y", colors=red, direction="in")
    objective_axis.tick_params(axis="x", direction="in")

    objective_axis.spines["left"].set_color(blue)
    constraint_axis.spines["right"].set_color(red)

    objective_axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    objective_axis.legend(lines, [l.get_label() for l in lines], loc="best", frameon=True, edgecolor="black")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    # 2. Cure Quality Plot (Voxel Count)
    undercured = np.asarray([record.undercured_voxels for record in history], dtype=np.int64)
    overcured = np.asarray([record.overcured_voxels for record in history], dtype=np.int64)

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    axis.plot(
    iterations,
    undercured,
    color=blue,
    linestyle="-",
    linewidth=1.8,
    marker="o",
    markevery=max(1, len(iterations) // 10),
    markersize=4,
    label="Undercured voxels",
    )   
    axis.plot(
    iterations,
    overcured,
    color=red,
    linestyle="--",
    linewidth=1.8,
    marker="s",
    markevery=max(1, len(iterations) // 10),
    markersize=4,
    label="Overcured voxels",
    )
    
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Voxel Count")
    axis.tick_params(direction="in")
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
    axis.legend(loc="best", frameon=True, edgecolor="black")

    figure.tight_layout()
    cure_output_path = output_path.with_name(f"{output_path.stem}_cure{output_path.suffix}")
    figure.savefig(cure_output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)



def _write_history_csv(history, output_path):
    """Write only the metrics needed for the optimization-history analysis."""
    fieldnames = [
        "iteration",
        "original_objective",
        "formulation_objective",
        "raw_constraint_violation",
        "undercured_voxels",
        "overcured_voxels",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(
                {
                    "iteration": record.iteration,
                    "original_objective": record.original_objective,
                    "formulation_objective": record.formulation_objective,
                    "raw_constraint_violation": record.raw_constraint_violation,
                    "undercured_voxels": record.undercured_voxels,
                    "overcured_voxels": record.overcured_voxels,
                }
            )


def _write_json(data, output_path):
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)


def _parse_overrides(values):
    parameters = {}
    for value in values or []:
        key, separator, raw_value = value.partition("=")
        if not separator or not key.strip() or not raw_value.strip():
            raise ValueError(f"Expected KEY=VALUE, got {value!r}.")
        key = key.strip()
        if key in parameters:
            raise ValueError(f"Parameter {key!r} was provided more than once.")
        parameters[key] = yaml.safe_load(raw_value)
    return parameters


def _finite_objective_and_gradient(objective, gradient):
    return bool(
        np.isfinite(float(objective))
        and np.all(np.isfinite(np.asarray(gradient)))
    )


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _default_output_directory(benchmark_config, problem_name, solver_name):
    return (
        _project_path(benchmark_config["output"]["directory"])
        / "optimization_history"
        / benchmark_config["benchmark"]["name"]
        / problem_name
        / solver_name
    )


def main() -> None:
    args = parse_arguments()
    run_history_analysis(
        benchmark_config_path=args.config,
        optimization_config_path=args.optimization_config,
        problem_name=args.problem,
        solver_name=args.solver,
        problem_parameters=_parse_overrides(args.problem_param),
        penalty_parameters=_parse_overrides(args.penalty_param),
        solver_parameters=_parse_overrides(args.solver_param),
        output_directory=args.output_dir,
        save_pdf=args.pdf,
    )


if __name__ == "__main__":
    main()
