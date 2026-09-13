"""Load defaults for problems, formulation adapters, and solvers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from scripts.optimization.formulations import (
    PENALTY_POLICIES,
    resolve_penalty_parameters,
)
from scripts.optimization.problems import PROBLEMS
from scripts.optimization.solvers import (
    SOLVERS,
    normalize_solver_name,
    resolve_solver_parameters,
)


DEFAULT_OPTIMIZATION_CONFIG_PATH = Path("configs/optimization_defaults.yaml")


def load_optimization_defaults(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Optimization config must be a mapping: {config_path}")

    config = deepcopy(raw_config)
    unknown_sections = set(config) - {"problems", "penalties", "solvers"}
    if unknown_sections:
        raise ValueError(
            "Unknown optimization config section(s): "
            + ", ".join(sorted(unknown_sections))
        )

    problems = config.setdefault("problems", {})
    penalties = config.setdefault("penalties", {})
    solvers = config.setdefault("solvers", {})
    if not isinstance(problems, dict):
        raise ValueError("optimization problems must be a mapping.")
    if not isinstance(solvers, dict):
        raise ValueError("optimization solvers must be a mapping.")
    if not isinstance(penalties, dict):
        raise ValueError("optimization penalties must be a mapping.")

    for problem_name, parameters in problems.items():
        if problem_name not in PROBLEMS:
            raise ValueError(
                f"Unknown problem {problem_name!r} in {config_path}. "
                f"Available: {', '.join(PROBLEMS)}"
            )
        if not isinstance(parameters, dict):
            raise ValueError(f"problems.{problem_name} must be a mapping.")

        PROBLEMS[problem_name].resolve_parameters(parameters)

    for problem_name, parameters in penalties.items():
        if problem_name not in PENALTY_POLICIES:
            raise ValueError(
                f"Problem {problem_name!r} has no configurable penalty adapter. "
                f"Available: {', '.join(PENALTY_POLICIES)}"
            )
        if not isinstance(parameters, dict):
            raise ValueError(f"penalties.{problem_name} must be a mapping.")
        resolve_penalty_parameters(problem_name, parameters)

    for solver_name, parameters in solvers.items():
        if solver_name not in SOLVERS:
            raise ValueError(
                f"Unknown solver {solver_name!r} in {config_path}. "
                f"Use canonical names: {', '.join(SOLVERS)}"
            )
        if not isinstance(parameters, dict):
            raise ValueError(f"solvers.{solver_name} must be a mapping.")
        resolve_solver_parameters(solver_name, parameters)

    return config


def defaults_for_selection(
    config: dict[str, Any],
    problem_name: str,
    solver_name: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    canonical_solver = normalize_solver_name(solver_name)
    return (
        dict(config["problems"].get(problem_name, {})),
        dict(config["penalties"].get(problem_name, {})),
        dict(config["solvers"].get(canonical_solver, {})),
    )
