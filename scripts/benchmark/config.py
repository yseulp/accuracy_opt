"""Loading and validation for multi-run benchmark YAML files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.benchmark.core import REQUIRED_METRICS, SUPPORTED_METRICS


DEFAULT_METRICS = [
    "undercured_voxels",
    "overcured_voxels",
    "outside_energy",
    "min_inside_energy",
    "runtime_seconds",
    "optimization_success",
    "converged",
    "iterations",
]

DEFAULT_OUTPUT = {
    "directory": "results",
    "comparison_csv": "comparison.csv",
    "save_intensity_arrays": True,
    "save_plots": True,
}

DEFAULT_VISUALIZATION = {
    "slice_y": None,
}

REQUIRED_PROCESS_KEYS = (
    "layer_height",
    "exposure_time",
    "atten_coef",
    "sigma",
    "I_max",
    "E_crit",
    "min_infl_factor",
)


def load_benchmark_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Benchmark config must be a mapping: {config_path}")

    config = deepcopy(raw_config)
    benchmark = _require_mapping(config, "benchmark")
    benchmark.setdefault("name", config_path.stem)
    benchmark.setdefault("random_seed", None)

    geometry = _require_mapping(config, "geometry")
    geometry.setdefault("type", "testcube")

    _require_mapping(config, "process")
    initialization = _require_mapping(config, "initialization")
    initialization.setdefault("type", "target_scaled")
    initialization.setdefault("scale", 1.0)

    target_energy = config.setdefault("target_energy", {})
    if not isinstance(target_energy, dict):
        raise ValueError("target_energy must be a mapping.")
    target_energy.setdefault("type", "attenuated_layers")
    target_energy.setdefault("reference", "initial_intensity")
    target_energy.setdefault("scale", 0.5)

    config.setdefault("metrics", deepcopy(DEFAULT_METRICS))
    output = config.setdefault("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping.")
    for key, value in DEFAULT_OUTPUT.items():
        output.setdefault(key, value)

    visualization = config.setdefault("visualization", {})
    if not isinstance(visualization, dict):
        raise ValueError("visualization must be a mapping.")
    for key, value in DEFAULT_VISUALIZATION.items():
        visualization.setdefault(key, value)

    validate_benchmark_config(config, config_path)
    return config


def validate_benchmark_config(
    config: dict[str, Any],
    config_path: Path | None = None,
) -> None:
    source = config_path if config_path is not None else Path("<runtime-config>")
    benchmark = _require_mapping(config, "benchmark")
    geometry = _require_mapping(config, "geometry")
    process = _require_mapping(config, "process")
    initialization = _require_mapping(config, "initialization")
    target_energy = _require_mapping(config, "target_energy")

    _reject_unknown_keys(
        config,
        {
            "benchmark",
            "geometry",
            "process",
            "initialization",
            "target_energy",
            "metrics",
            "visualization",
            "output",
        },
        "top-level config",
    )
    _reject_unknown_keys(benchmark, {"name", "random_seed"}, "benchmark")
    _reject_unknown_keys(
        geometry, {"type", "domain_dim", "offset"}, "geometry"
    )
    _reject_unknown_keys(initialization, {"type", "scale"}, "initialization")
    _reject_unknown_keys(
        target_energy, {"type", "reference", "scale"}, "target_energy"
    )

    if not isinstance(benchmark.get("name"), str) or not benchmark["name"]:
        raise ValueError(f"benchmark.name must be a non-empty string in {source}")
    _validate_path_segment(benchmark["name"], "benchmark.name")
    seed = benchmark.get("random_seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("benchmark.random_seed must be an integer or null.")

    if geometry.get("type") != "testcube":
        raise ValueError("Only geometry.type='testcube' is supported.")
    domain_dim = geometry.get("domain_dim")
    offset = geometry.get("offset")
    if not _is_triplet(domain_dim, positive=True):
        raise ValueError("geometry.domain_dim must contain three positive integers.")
    if not _is_triplet(offset, positive=False):
        raise ValueError("geometry.offset must contain three non-negative integers.")
    inner_sizes = [dim - 2 * off for dim, off in zip(domain_dim, offset, strict=True)]
    if any(size <= 0 for size in inner_sizes):
        raise ValueError("geometry.offset leaves no interior testcube voxels.")
    if int(np.prod(inner_sizes)) == int(np.prod(domain_dim)):
        raise ValueError("geometry must contain at least one outside voxel.")

    unknown_process = set(process) - set(REQUIRED_PROCESS_KEYS)
    if unknown_process:
        raise ValueError(
            "Unknown process parameter(s): " + ", ".join(sorted(unknown_process))
        )
    for key in REQUIRED_PROCESS_KEYS:
        if key not in process:
            raise ValueError(f"Missing process.{key} in {source}")
        process[key] = _positive_float(process[key], f"process.{key}")
    if process["min_infl_factor"] >= 1:
        raise ValueError("process.min_infl_factor must be smaller than one.")

    if initialization.get("type") != "target_scaled":
        raise ValueError("Only initialization.type='target_scaled' is supported.")
    initialization["scale"] = _finite_float(
        initialization.get("scale"), "initialization.scale"
    )
    if not 0 <= initialization["scale"] <= 1:
        raise ValueError("initialization.scale must be between zero and one.")

    if target_energy.get("type") != "attenuated_layers":
        raise ValueError("Only target_energy.type='attenuated_layers' is supported.")
    if target_energy.get("reference") != "initial_intensity":
        raise ValueError(
            "Only target_energy.reference='initial_intensity' is supported."
        )
    target_energy["scale"] = _positive_float(
        target_energy.get("scale"), "target_energy.scale"
    )

    metrics = config.get("metrics")
    if not isinstance(metrics, list) or not metrics or not all(
        isinstance(metric, str) and metric for metric in metrics
    ):
        raise ValueError("metrics must be a non-empty list of metric names.")
    if len(metrics) != len(set(metrics)):
        raise ValueError("metrics must not contain duplicates.")
    unknown_metrics = set(metrics) - SUPPORTED_METRICS
    if unknown_metrics:
        raise ValueError("Unknown metric(s): " + ", ".join(sorted(unknown_metrics)))
    missing_metrics = REQUIRED_METRICS - set(metrics)
    if missing_metrics:
        raise ValueError(
            "Missing required common metric(s): "
            + ", ".join(sorted(missing_metrics))
        )

    output = _require_mapping(config, "output")
    _reject_unknown_keys(
        output,
        {
            "directory",
            "comparison_csv",
            "save_intensity_arrays",
            "save_plots",
        },
        "output",
    )
    if not isinstance(output.get("directory"), str) or not output["directory"]:
        raise ValueError("output.directory must be a non-empty path string.")
    if not isinstance(output.get("comparison_csv"), str) or not output["comparison_csv"]:
        raise ValueError("output.comparison_csv must be a non-empty path string.")
    comparison_path = Path(output["comparison_csv"])
    if (
        comparison_path.is_absolute()
        or len(comparison_path.parts) != 1
        or comparison_path.suffix.lower() != ".csv"
    ):
        raise ValueError("output.comparison_csv must be one relative .csv filename.")
    for key in ("save_intensity_arrays", "save_plots"):
        if not isinstance(output.get(key), bool):
            raise ValueError(f"output.{key} must be true or false.")

    visualization = _require_mapping(config, "visualization")
    unknown_visualization = set(visualization) - {"slice_y"}
    if unknown_visualization:
        raise ValueError(
            "Unknown visualization setting(s): "
            + ", ".join(sorted(unknown_visualization))
        )
    slice_y = visualization.get("slice_y")
    if slice_y is None:
        visualization["slice_y"] = domain_dim[1] // 2
    elif (
        isinstance(slice_y, bool)
        or not isinstance(slice_y, int)
        or not 0 <= slice_y < domain_dim[1]
    ):
        raise ValueError(
            f"visualization.slice_y must be between 0 and {domain_dim[1] - 1}."
        )


def write_config_snapshot(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(config, output_file, sort_keys=False)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid '{key}' section in benchmark config.")
    return value


def _reject_unknown_keys(
    mapping: dict[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {section} field(s): " + ", ".join(sorted(unknown))
        )


def _validate_path_segment(value: str, name: str) -> None:
    path = Path(value)
    if value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError(f"{name} must be one safe path segment.")


def _is_triplet(values: Any, *, positive: bool) -> bool:
    if not isinstance(values, list) or len(values) != 3:
        return False
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if positive and value <= 0:
            return False
        if not positive and value < 0:
            return False
    return True


def _positive_float(value: Any, name: str) -> float:
    resolved = _finite_float(value, name)
    if resolved <= 0:
        raise ValueError(f"{name} must be greater than zero.")
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
