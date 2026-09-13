"""Sweep exterior-penalty weights while logging objective components per step."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.benchmark.config import load_benchmark_config
from scripts.benchmark.core import BenchmarkData, build_benchmark
from scripts.model.simulation import simulate
from scripts.optimization.formulations import (
    FORMULATION_ADAPTERS,
    PENALTY_POLICIES,
    resolve_penalty_parameters,
)
from scripts.optimization.problems import PROBLEMS
from scripts.optimization.solvers import (
    normalize_solver_name,
    resolve_solver_parameters,
)


DEFAULT_STUDY_CONFIG = Path("configs/studies/penalty_study_guven.yaml")
PGD_SOLVER = "projected_gradient_descent"


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    penalty_weight: float
    original_objective: float
    raw_penalty: float
    weighted_penalty: float
    penalized_objective: float
    gradient_norm: float
    undercured_voxels: int
    overcured_voxels: int


@dataclass(frozen=True)
class SweepResult:
    penalty_weight: float
    final_original_objective: float
    final_constraint_violation: float
    final_weighted_penalty: float
    final_penalized_objective: float
    final_gradient_norm: float
    undercured_voxels: int
    overcured_voxels: int
    iterations_completed: int
    runtime_seconds: float
    logged_runtime_seconds: float
    success: bool


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study the effect of exterior-penalty weights on PGD."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_STUDY_CONFIG,
        help="Path to the penalty-study YAML configuration.",
    )
    return parser.parse_args()


def load_study_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Penalty-study config must be a mapping: {config_path}")

    allowed = {
        "benchmark_config",
        "problem",
        "solver",
        "problem_parameters",
        "solver_parameters",
        "penalty_weights",
        "convergence_plot_weights",
        "output_directory",
    }
    unknown = set(raw_config) - allowed
    if unknown:
        raise ValueError(
            "Unknown penalty-study setting(s): " + ", ".join(sorted(unknown))
        )

    config = dict(raw_config)
    for key in ("benchmark_config", "problem", "penalty_weights", "output_directory"):
        if key not in config:
            raise ValueError(f"Missing required penalty-study setting: {key}")

    problem_name = config["problem"]
    if not isinstance(problem_name, str) or not problem_name:
        raise ValueError("problem must be a non-empty string.")
    if problem_name not in PROBLEMS:
        raise ValueError(
            f"Unknown problem {problem_name!r}. Available: {', '.join(PROBLEMS)}"
        )
    if problem_name not in PENALTY_POLICIES:
        raise ValueError(
            f"Problem {problem_name!r} has no exterior-penalty policy to study."
        )

    configured_solver = config.get("solver", PGD_SOLVER)
    if not isinstance(configured_solver, str) or not configured_solver:
        raise ValueError("solver must be a non-empty string.")
    solver_name = normalize_solver_name(configured_solver)
    if solver_name != PGD_SOLVER:
        raise ValueError("The penalty study currently supports only projected_gradient_descent.")
    config["solver"] = solver_name

    problem_parameters = config.get("problem_parameters", {})
    solver_parameters = config.get("solver_parameters", {})
    if not isinstance(problem_parameters, dict):
        raise ValueError("problem_parameters must be a mapping.")
    if not isinstance(solver_parameters, dict):
        raise ValueError("solver_parameters must be a mapping.")
    config["problem_parameters"] = PROBLEMS[problem_name].resolve_parameters(
        problem_parameters
    )
    config["solver_parameters"] = resolve_solver_parameters(
        solver_name, solver_parameters
    )
    if config["solver_parameters"]["objective_tolerance"] is not None:
        raise ValueError(
            "solver_parameters.objective_tolerance must be null so every sweep "
            "run completes the same number of iterations."
        )

    config["penalty_weights"] = _positive_unique_weights(
        config["penalty_weights"], "penalty_weights"
    )
    selected_weights = config.get(
        "convergence_plot_weights", config["penalty_weights"]
    )
    selected_weights = _positive_unique_weights(
        selected_weights, "convergence_plot_weights"
    )
    unavailable = [
        weight
        for weight in selected_weights
        if weight not in config["penalty_weights"]
    ]
    if unavailable:
        raise ValueError(
            "convergence_plot_weights must be contained in penalty_weights: "
            + ", ".join(str(weight) for weight in unavailable)
        )
    config["convergence_plot_weights"] = selected_weights

    if not isinstance(config["benchmark_config"], str) or not config[
        "benchmark_config"
    ].strip():
        raise ValueError("benchmark_config must be a non-empty path string.")
    if not isinstance(config["output_directory"], str) or not config[
        "output_directory"
    ].strip():
        raise ValueError("output_directory must be a non-empty path string.")
    return config


def run_study(config_path: Path) -> tuple[list[SweepResult], dict[float, list[IterationRecord]]]:
    config_path = config_path.resolve()
    study_config = load_study_config(config_path)
    benchmark_path = _resolve_project_path(study_config["benchmark_config"])
    output_directory = _resolve_project_path(study_config["output_directory"])
    benchmark_config = load_benchmark_config(benchmark_path)
    _set_random_seed(benchmark_config["benchmark"]["random_seed"])
    benchmark = build_benchmark(benchmark_config)

    problem = PROBLEMS[study_config["problem"]]
    penalty_policy = PENALTY_POLICIES[problem.name]
    penalty_adapter = FORMULATION_ADAPTERS["penalty"]
    if not penalty_adapter.supports(problem):
        raise ValueError(f"Penalty adapter does not support problem {problem.name!r}.")

    output_directory.mkdir(parents=True, exist_ok=True)
    histories: dict[float, list[IterationRecord]] = {}
    summaries: list[SweepResult] = []
    for penalty_weight in study_config["penalty_weights"]:
        print(f"Running {problem.name} with penalty_weight={penalty_weight:g}")
        penalty_parameters = resolve_penalty_parameters(
            problem.name, {"penalty_weight": penalty_weight}
        )
        formulation = penalty_adapter.build(
            problem,
            I_start=benchmark.I_start,
            T=benchmark.T,
            C=benchmark.C,
            kernel=benchmark.kernel,
            domain_dim=benchmark.domain_dim,
            param=benchmark.param,
            problem_parameters=study_config["problem_parameters"],
            penalty_parameters=penalty_parameters,
        )
        history, summary = run_pgd_with_history(
            benchmark=benchmark,
            problem=problem,
            formulation=formulation,
            problem_parameters=study_config["problem_parameters"],
            penalty_weight=penalty_weight,
            penalty_power=penalty_policy.power,
            step_size=study_config["solver_parameters"]["step_size"],
            max_iterations=study_config["solver_parameters"]["max_iterations"],
        )
        histories[penalty_weight] = history
        summaries.append(summary)
        _write_csv(
            [asdict(record) for record in history],
            output_directory / f"convergence_penalty_{_weight_slug(penalty_weight)}.csv",
        )

    summaries.sort(key=lambda result: result.penalty_weight)
    _write_csv(
        [asdict(summary) for summary in summaries],
        output_directory / "penalty_study.csv",
    )
    _write_resolved_config(
        study_config,
        benchmark_path,
        benchmark_config,
        config_path,
        output_directory / "resolved_study_config.yaml",
    )
    create_summary_plots(summaries, output_directory)
    create_convergence_plot(
        histories,
        study_config["convergence_plot_weights"],
        output_directory / "convergence_comparison.png",
    )
    print(f"Penalty study saved in: {output_directory}")
    return summaries, histories


def run_pgd_with_history(
    *,
    benchmark: BenchmarkData,
    problem,
    formulation,
    problem_parameters: dict[str, Any],
    penalty_weight: float,
    penalty_power: int,
    step_size: float,
    max_iterations: int,
) -> tuple[list[IterationRecord], SweepResult]:
    lower_bounds = jnp.asarray([bound[0] for bound in formulation.bounds])
    upper_bounds = jnp.asarray([bound[1] for bound in formulation.bounds])
    intensity = jnp.clip(benchmark.I_start, lower_bounds, upper_bounds)
    value_and_grad = jax.value_and_grad(formulation.objective)
    history: list[IterationRecord] = []
    success = True
    logged_start_time = time.perf_counter()
    optimization_runtime_seconds = 0.0

    record, gradient, evaluation_seconds = _evaluate_iteration(
        iteration=0,
        intensity=intensity,
        benchmark=benchmark,
        problem=problem,
        problem_parameters=problem_parameters,
        penalty_weight=penalty_weight,
        penalty_power=penalty_power,
        value_and_grad=value_and_grad,
    )
    optimization_runtime_seconds += evaluation_seconds
    history.append(record)

    for iteration in range(1, max_iterations + 1):
        if not _finite_record_and_gradient(record, gradient):
            success = False
            break
        update_start_time = time.perf_counter()
        candidate = jnp.clip(
            intensity - step_size * gradient,
            lower_bounds,
            upper_bounds,
        )
        candidate = jax.block_until_ready(candidate)
        optimization_runtime_seconds += time.perf_counter() - update_start_time
        candidate_record, candidate_gradient, evaluation_seconds = _evaluate_iteration(
            iteration=iteration,
            intensity=candidate,
            benchmark=benchmark,
            problem=problem,
            problem_parameters=problem_parameters,
            penalty_weight=penalty_weight,
            penalty_power=penalty_power,
            value_and_grad=value_and_grad,
        )
        optimization_runtime_seconds += evaluation_seconds
        if not _finite_record_and_gradient(candidate_record, candidate_gradient):
            success = False
            break
        intensity = candidate
        record = candidate_record
        gradient = candidate_gradient
        history.append(record)

    if not _finite_record_and_gradient(record, gradient):
        success = False
    logged_runtime_seconds = time.perf_counter() - logged_start_time
    final = history[-1]
    return history, SweepResult(
        penalty_weight=penalty_weight,
        final_original_objective=final.original_objective,
        final_constraint_violation=final.raw_penalty,
        final_weighted_penalty=final.weighted_penalty,
        final_penalized_objective=final.penalized_objective,
        final_gradient_norm=final.gradient_norm,
        undercured_voxels=final.undercured_voxels,
        overcured_voxels=final.overcured_voxels,
        iterations_completed=final.iteration,
        runtime_seconds=optimization_runtime_seconds,
        logged_runtime_seconds=logged_runtime_seconds,
        success=success and final.iteration == max_iterations,
    )


def _evaluate_iteration(
    *,
    iteration,
    intensity,
    benchmark,
    problem,
    problem_parameters,
    penalty_weight,
    penalty_power,
    value_and_grad,
):
    evaluation_start_time = time.perf_counter()
    penalized_objective, gradient = value_and_grad(intensity)
    penalized_objective, gradient = jax.block_until_ready(
        (penalized_objective, gradient)
    )
    evaluation_seconds = time.perf_counter() - evaluation_start_time
    original_objective = problem.objective(
        intensity,
        benchmark.T,
        benchmark.C,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
        problem_parameters,
    )
    residuals = problem.constraint_residuals(
        intensity,
        benchmark.T,
        benchmark.C,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
        problem_parameters,
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
    raw_penalty = float(jnp.sum(jnp.maximum(0.0, -residuals) ** penalty_power))
    weighted_penalty = penalty_weight * raw_penalty
    original_objective = float(original_objective)
    penalized_objective = float(penalized_objective)
    expected_penalized = original_objective + weighted_penalty
    if np.isfinite(expected_penalized) and not np.isclose(
        penalized_objective, expected_penalized, rtol=1e-9, atol=1e-10
    ):
        raise RuntimeError(
            "Penalty adapter value does not equal base objective plus weighted penalty."
        )

    energy_np = np.asarray(energy, dtype=np.float64)
    C_np = np.asarray(benchmark.C)
    record = IterationRecord(
        iteration=iteration,
        penalty_weight=penalty_weight,
        original_objective=original_objective,
        raw_penalty=raw_penalty,
        weighted_penalty=weighted_penalty,
        penalized_objective=penalized_objective,
        gradient_norm=float(jnp.linalg.norm(gradient)),
        undercured_voxels=int(
            np.sum(energy_np[C_np == 1] < float(benchmark.param.E_crit))
        ),
        overcured_voxels=int(
            np.sum(energy_np[C_np == 0] >= float(benchmark.param.E_crit))
        ),
    )
    return record, gradient, evaluation_seconds


def create_summary_plots(
    summaries: list[SweepResult],
    output_directory: Path,
) -> None:
    weights = np.asarray([result.penalty_weight for result in summaries])

    _single_summary_plot(
        weights,
        np.asarray([result.final_original_objective for result in summaries]),
        ylabel="Final original objective",
        title="Original objective vs. penalty weight",
        output_path=output_directory / "objective_vs_penalty.png",
    )
    _single_summary_plot(
        weights,
        np.asarray([result.final_constraint_violation for result in summaries]),
        ylabel="Final raw penalty P(I)",
        title="Raw penalty vs. penalty weight",
        output_path=output_directory / "constraint_vs_penalty.png",
    )

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(
        weights,
        [result.undercured_voxels for result in summaries],
        marker="o",
        label="Undercured voxels",
    )
    axis.plot(
        weights,
        [result.overcured_voxels for result in summaries],
        marker="s",
        label="Overcured voxels",
    )
    axis.set_xscale("log")
    axis.set_xlabel("Penalty weight")
    axis.set_ylabel("Voxel count")
    axis.set_title("Cure violations vs. penalty weight")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_directory / "cure_violations_vs_penalty.png", dpi=160)
    plt.close(figure)


def create_convergence_plot(
    histories: dict[float, list[IterationRecord]],
    selected_weights: list[float],
    output_path: Path,
) -> None:
    column_count = 2 if len(selected_weights) > 1 else 1
    row_count = math.ceil(len(selected_weights) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.2 * column_count, 4.2 * row_count),
        squeeze=False,
        sharex=True,
    )
    selected_histories = [histories[weight] for weight in selected_weights]
    objective_values = [
        record.original_objective
        for history in selected_histories
        for record in history
    ]
    violation_values = [
        record.raw_penalty
        for history in selected_histories
        for record in history
    ]
    objective_limits = _plot_limits(objective_values)
    violation_limits = _plot_limits(violation_values)

    for axis, penalty_weight, history in zip(
        axes.flat, selected_weights, selected_histories, strict=False
    ):
        iterations = [record.iteration for record in history]
        objective_axis = axis
        violation_axis = objective_axis.twinx()
        objective_line = objective_axis.plot(
            iterations,
            [record.original_objective for record in history],
            color="tab:blue",
            label="Original objective",
        )[0]
        violation_line = violation_axis.plot(
            iterations,
            [record.raw_penalty for record in history],
            color="tab:red",
            label="Constraint violation",
        )[0]
        objective_axis.set_yscale("symlog")
        violation_axis.set_yscale("symlog")
        objective_axis.set_ylim(objective_limits)
        violation_axis.set_ylim(violation_limits)
        objective_axis.set_xlabel("Iteration")
        objective_axis.set_ylabel("Original objective", color="tab:blue")
        violation_axis.set_ylabel("Raw penalty P(I)", color="tab:red")
        objective_axis.set_title(f"penalty_weight = {penalty_weight:g}")
        objective_axis.grid(True, alpha=0.25)
        objective_axis.legend(
            [objective_line, violation_line],
            [objective_line.get_label(), violation_line.get_label()],
            loc="best",
        )

    for axis in axes.flat[len(selected_weights):]:
        axis.set_visible(False)
    figure.suptitle("Penalty-study convergence", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _single_summary_plot(weights, values, *, ylabel, title, output_path):
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(weights, values, marker="o")
    axis.set_xscale("log")
    axis.set_xlabel("Penalty weight")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_limits(values):
    finite = np.asarray([value for value in values if np.isfinite(value)])
    if finite.size == 0:
        return (-1.0, 1.0)
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1e-9)
        return minimum - padding, maximum + padding
    padding = 0.05 * (maximum - minimum)
    return minimum - padding, maximum + padding


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_resolved_config(
    study_config,
    benchmark_path,
    benchmark_config,
    study_config_path,
    output_path,
):
    resolved = dict(study_config)
    resolved["study_config_source"] = str(study_config_path)
    resolved["benchmark_config"] = str(benchmark_path)
    resolved["benchmark_config_sha256"] = hashlib.sha256(
        json.dumps(benchmark_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    resolved["resolved_benchmark_config"] = benchmark_config
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(resolved, output_file, sort_keys=False)


def _positive_unique_weights(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    weights = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{name} must contain positive finite numbers.")
        try:
            weight = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must contain positive finite numbers."
            ) from error
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name} must contain positive finite numbers.")
        if weight in weights:
            raise ValueError(f"{name} must not contain duplicate values.")
        weights.append(weight)
    return weights


def _finite_record_and_gradient(record, gradient):
    numeric_values = (
        record.original_objective,
        record.raw_penalty,
        record.weighted_penalty,
        record.penalized_objective,
        record.gradient_norm,
    )
    return all(np.isfinite(value) for value in numeric_values) and np.all(
        np.isfinite(np.asarray(gradient))
    )


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _weight_slug(weight: float) -> str:
    if weight.is_integer():
        return str(int(weight))
    return repr(weight).replace("+", "")


def _set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_arguments()
    run_study(args.config)


if __name__ == "__main__":
    main()
