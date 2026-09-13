"""Combine a benchmark config with selected problems and one solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from copy import copy, deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import yaml

from scripts.benchmark.config import load_benchmark_config, write_config_snapshot
from scripts.benchmark.core import BenchmarkData, build_benchmark, evaluate_solution
from scripts.optimization.config import (
    DEFAULT_OPTIMIZATION_CONFIG_PATH,
    defaults_for_selection,
    load_optimization_defaults,
)
from scripts.optimization.problems import PROBLEMS
from scripts.optimization.solvers import (
    SOLVERS,
    SOLVER_ALIASES,
    check_compatibility,
    resolve_selection,
    run_solver,
)
from scripts.model.simulation import cured, simulate
from scripts.visualization.plots import plot_xz


DEFAULT_CONFIG_PATH = Path("configs/benchmark_testcube_20.yaml")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine a shared benchmark configuration with one or more "
            "optimization problems and one solver."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML benchmark configuration.",
    )
    parser.add_argument(
        "--problem",
        required=True,
        nargs="+",
        metavar="PROBLEM",
        help=(
            "One or more problems, separated by spaces or commas. Use 'all' "
            "for every problem compatible with the selected solver."
        ),
    )
    parser.add_argument(
        "--optimization-config",
        type=Path,
        default=DEFAULT_OPTIMIZATION_CONFIG_PATH,
        help="Path to configurable problem, penalty, and solver defaults.",
    )
    parser.add_argument(
        "--solver",
        required=True,
        choices=sorted(set(SOLVERS) | set(SOLVER_ALIASES)),
        help="Solver used for all selected problems.",
    )
    parser.add_argument(
        "--problem-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override a relevant problem parameter. Repeat for multiple values."
        ),
    )
    parser.add_argument(
        "--penalty-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override an exterior-penalty adapter parameter. This is only "
            "valid when the selected solver requires a penalty."
        ),
    )
    parser.add_argument(
        "--solver-param",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a relevant solver parameter. Repeat for multiple values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output.directory from the benchmark configuration.",
    )
    return parser.parse_args()


def compute_config_sha256(config: dict[str, Any]) -> str:
    normalized = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def parse_parameter_overrides(
    values: list[str] | None,
    option_name: str,
) -> dict[str, Any]:
    parameters = {}
    for value in values or []:
        key, separator, raw_value = value.partition("=")
        key = key.strip()
        if not separator or not key or not raw_value.strip():
            raise ValueError(
                f"Invalid {option_name} {value!r}. Expected KEY=VALUE."
            )
        if key in parameters:
            raise ValueError(f"Parameter {key!r} was provided more than once.")
        parameters[key] = yaml.safe_load(raw_value)
    return parameters


def resolve_problem_names(values: list[str], solver_name: str) -> list[str]:
    names = [
        name.strip()
        for value in values
        for name in value.split(",")
        if name.strip()
    ]
    if not names:
        raise ValueError("--problem requires at least one problem name.")
    if "all" in names:
        if len(names) != 1:
            raise ValueError("Use --problem all without additional problem names.")
        compatible = []
        for problem_name in PROBLEMS:
            try:
                check_compatibility(problem_name, solver_name)
            except ValueError:
                continue
            compatible.append(problem_name)
        if not compatible:
            raise ValueError(f"No problems are compatible with solver {solver_name!r}.")
        return compatible

    unknown = [name for name in names if name not in PROBLEMS]
    if unknown:
        raise ValueError(
            "Unknown problem(s): "
            + ", ".join(unknown)
            + f". Available: {', '.join(PROBLEMS)}"
        )
    if len(names) != len(set(names)):
        raise ValueError("Each problem may be selected only once.")
    for problem_name in names:
        check_compatibility(problem_name, solver_name)
    return names


def save_plot_figure(figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_shared_plots(
    benchmark: BenchmarkData,
    output_dir: Path,
    slice_y: int,
    energy_color_limits: tuple[float, float],
) -> None:
    initial_energy = simulate(
        benchmark.I_start,
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
    )
    initial_cured = cured(initial_energy, benchmark.param)

    save_plot_figure(
        plot_xz(
            initial_energy,
            benchmark.domain_dim,
            "energy unoptimized",
            color_limits=energy_color_limits,
            slice_y=slice_y,
        ),
        output_dir / "initial_energy.png",
    )
    save_plot_figure(
        plot_xz(
            initial_cured,
            benchmark.domain_dim,
            "cured unoptimized",
            slice_y=slice_y,
        ),
        output_dir / "initial_cured.png",
    )
    save_plot_figure(
        plot_xz(
            benchmark.C,
            benchmark.domain_dim,
            "target geometry",
            slice_y=slice_y,
        ),
        output_dir / "target_geometry.png",
    )


def save_run_plots(
    *,
    benchmark: BenchmarkData,
    I_opt: jnp.ndarray,
    optimized_energy: jnp.ndarray,
    output_dir: Path,
    title_suffix: str,
    slice_y: int,
    energy_color_limits: tuple[float, float],
) -> None:
    optimized_cured = cured(optimized_energy, benchmark.param)
    save_plot_figure(
        plot_xz(
            optimized_energy,
            benchmark.domain_dim,
            f"energy optimized ({title_suffix})",
            color_limits=energy_color_limits,
            slice_y=slice_y,
        ),
        output_dir / "optimized_energy.png",
    )
    save_plot_figure(
        plot_xz(
            optimized_cured,
            benchmark.domain_dim,
            f"cured optimized ({title_suffix})",
            slice_y=slice_y,
        ),
        output_dir / "optimized_cured.png",
    )
    save_plot_figure(
        plot_xz(
            I_opt,
            benchmark.domain_dim,
            f"optimized intensity ({title_suffix})",
            color_limits=(0.0, float(benchmark.param.I_max)),
            slice_y=slice_y,
        ),
        output_dir / "optimized_intensity.png",
    )


def build_experiment_snapshot(
    config: dict[str, Any],
    selection: dict[str, Any],
    optimization_config_path: Path,
) -> dict[str, Any]:
    snapshot = deepcopy(config)
    snapshot["optimization_defaults_source"] = str(
        optimization_config_path.resolve()
    )
    snapshot["selection"] = deepcopy(selection)
    return snapshot


def make_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2, allow_nan=False)
    temporary_path.replace(output_path)


def save_comparison_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


def print_comparison_table(results: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 120)
    print("FINAL PHYSICAL COMPARISON")
    print("=" * 120)
    header = (
        f"{'Problem':<18}{'Solver':<30}{'Success':>9}{'Runtime [s]':>14}"
        f"{'Iterations':>12}{'Undercured':>14}{'Overcured':>14}"
        f"{'Outside E':>15}{'Min inside E':>16}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        iterations = result.get("iterations")
        print(
            f"{result['problem']:<18}{result['solver']:<30}"
            f"{str(result['optimization_success']):>9}"
            f"{result['runtime_seconds']:>14.4f}"
            f"{str(iterations) if iterations is not None else '-':>12}"
            f"{result['undercured_voxels']:>14d}"
            f"{result['overcured_voxels']:>14d}"
            f"{result['outside_energy']:>15.6g}"
            f"{result['min_inside_energy']:>16.6g}"
        )
    print("\nObjective values use different formulations and are not comparable.")


def execute_selection(
    *,
    selection: dict[str, Any],
    benchmark: BenchmarkData,
    config: dict[str, Any],
    config_path: Path,
    optimization_config_path: Path,
    benchmark_output: Path,
    energy_color_limits: tuple[float, float],
) -> dict[str, Any]:
    problem_name = selection["problem"]
    solver_name = selection["solver"]
    run_output = benchmark_output / problem_name / solver_name

    print("\n" + "=" * 72)
    print(f"Problem: {problem_name}")
    print(f"Solver: {solver_name}")
    print(f"Formulation: {selection['formulation']}")
    print(f"Constraint handling: {selection['constraint_handling']}")
    print("=" * 72)

    experiment_snapshot = build_experiment_snapshot(
        config,
        selection,
        optimization_config_path,
    )
    config_sha256 = compute_config_sha256(experiment_snapshot)
    write_config_snapshot(
        experiment_snapshot,
        run_output / "resolved_config.yaml",
    )

    run_param = copy(benchmark.param)
    optimization = run_solver(
        solver_name,
        problem_name,
        I_start=jnp.array(benchmark.I_start),
        T=benchmark.T,
        C=benchmark.C,
        kernel=benchmark.kernel,
        domain_dim=benchmark.domain_dim,
        param=run_param,
        problem_parameters=selection["problem_parameters"],
        penalty_parameters=selection["penalty_parameters"],
        solver_parameters=selection["solver_parameters"],
        adapter_name=selection["adapter"],
    )
    metrics, optimized_energy = evaluate_solution(
        benchmark,
        optimization,
        config["metrics"],
    )
    base_objective = PROBLEMS[problem_name].objective(
        optimization.I_opt,
        benchmark.T,
        benchmark.C,
        benchmark.kernel,
        benchmark.domain_dim,
        run_param,
        selection["problem_parameters"],
    )
    base_objective = float(jax.block_until_ready(base_objective))
    result = make_serializable({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": config["benchmark"]["name"],
        "problem": problem_name,
        "solver": solver_name,
        "formulation": optimization.formulation,
        "adapter": selection["adapter"],
        "constraint_handling": selection["constraint_handling"],
        "problem_parameters": selection["problem_parameters"],
        "penalty_parameters": selection["penalty_parameters"],
        "solver_parameters": selection["solver_parameters"],
        "config_source_path": str(config_path.resolve()),
        "optimization_config_source_path": str(
            optimization_config_path.resolve()
        ),
        "config_sha256": config_sha256,
        "optimization_message": optimization.message,
        "initial_objective": optimization.initial_objective,
        "objective_value": optimization.final_objective,
        "formulation_objective_value": optimization.final_objective,
        "base_objective_value": base_objective,
        **metrics,
    })
    write_json(result, run_output / "result.json")

    if config["output"]["save_intensity_arrays"]:
        np.save(
            run_output / "optimized_intensity.npy",
            np.asarray(optimization.I_opt, dtype=np.float64),
        )
    if config["output"]["save_plots"]:
        save_run_plots(
            benchmark=benchmark,
            I_opt=optimization.I_opt,
            optimized_energy=optimized_energy,
            output_dir=run_output / "plots",
            title_suffix=f"{problem_name} + {solver_name}",
            slice_y=config["visualization"]["slice_y"],
            energy_color_limits=energy_color_limits,
        )

    print(f"  success:           {optimization.success}")
    print(f"  base objective:    {base_objective}")
    print(f"  formulation value: {optimization.final_objective}")
    print(f"  runtime:           {optimization.runtime_seconds:.4f} s")
    print(f"  undercured voxels: {metrics['undercured_voxels']}")
    print(f"  overcured voxels:  {metrics['overcured_voxels']}")
    print(f"  result:            {(run_output / 'result.json').resolve()}")
    return result


def main() -> None:
    args = parse_arguments()
    config = load_benchmark_config(args.config)
    optimization_defaults = load_optimization_defaults(args.optimization_config)
    if args.output_dir is not None:
        config["output"]["directory"] = str(args.output_dir)

    problem_names = resolve_problem_names(args.problem, args.solver)
    problem_overrides = parse_parameter_overrides(
        args.problem_param, "--problem-param"
    )
    solver_overrides = parse_parameter_overrides(
        args.solver_param, "--solver-param"
    )
    penalty_overrides = parse_parameter_overrides(
        args.penalty_param, "--penalty-param"
    )
    selections = []
    for problem_name in problem_names:
        problem_defaults, penalty_defaults, solver_defaults = defaults_for_selection(
            optimization_defaults,
            problem_name,
            args.solver,
        )
        selections.append(
            resolve_selection(
                problem_name,
                args.solver,
                problem_defaults=problem_defaults,
                penalty_defaults=penalty_defaults,
                solver_defaults=solver_defaults,
                problem_parameters=problem_overrides,
                penalty_parameters=penalty_overrides,
                solver_parameters=solver_overrides,
            )
        )

    set_random_seed(config["benchmark"]["random_seed"])
    benchmark = build_benchmark(config)
    benchmark_name = config["benchmark"]["name"]
    benchmark_output = Path(config["output"]["directory"]) / benchmark_name
    benchmark_output.mkdir(parents=True, exist_ok=True)

    energy_color_limits = (0.0, 0.0)
    if config["output"]["save_plots"]:
        max_energy = simulate(
            jnp.full_like(benchmark.I_start, float(benchmark.param.I_max)),
            benchmark.kernel,
            benchmark.domain_dim,
            benchmark.param,
        )
        energy_color_limits = (0.0, float(jnp.max(max_energy)))
        save_shared_plots(
            benchmark,
            benchmark_output / "shared",
            config["visualization"]["slice_y"],
            energy_color_limits,
        )

    print("=== Optimization benchmark ===")
    print(f"Benchmark: {benchmark_name}")
    print(f"Config: {args.config.resolve()}")
    print(f"Problems: {', '.join(problem_names)}")
    print(f"Solver: {selections[0]['solver']}")
    print(f"Domain dimensions: {benchmark.domain_dim}")

    results = [
        execute_selection(
            selection=selection,
            benchmark=benchmark,
            config=config,
            config_path=args.config,
            optimization_config_path=args.optimization_config,
            benchmark_output=benchmark_output,
            energy_color_limits=energy_color_limits,
        )
        for selection in selections
    ]

    csv_path = benchmark_output / config["output"]["comparison_csv"]
    save_comparison_csv(results, csv_path)
    print_comparison_table(results)
    print(f"\nComparison CSV: {csv_path.resolve()}")
    print(f"Artifacts: {benchmark_output.resolve()}")


if __name__ == "__main__":
    main()
