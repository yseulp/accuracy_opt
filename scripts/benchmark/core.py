"""Shared benchmark construction and solver-independent evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from scripts.model.geometry import testcube
from scripts.model.kernel import Kernel
from scripts.model.parameters import Param
from scripts.model.simulation import simulate

if TYPE_CHECKING:
    from scripts.optimization.solvers import OptimizationResult


SUPPORTED_METRICS = frozenset(
    {
        "undercured_voxels",
        "overcured_voxels",
        "outside_energy",
        "min_inside_energy",
        "max_inside_energy",
        "min_outside_energy",
        "max_outside_energy",
        "mean_inside_energy",
        "mean_outside_energy",
        "intensity_min",
        "intensity_max",
        "intensity_mean",
        "runtime_seconds",
        "optimization_success",
        "converged",
        "iterations",
        "gradient_norm",
    }
)

REQUIRED_METRICS = frozenset(
    {
        "undercured_voxels",
        "overcured_voxels",
        "outside_energy",
        "min_inside_energy",
        "runtime_seconds",
        "optimization_success",
    }
)


@dataclass(frozen=True)
class BenchmarkData:
    I_start: jnp.ndarray
    T: jnp.ndarray
    C: jnp.ndarray
    kernel: Any
    domain_dim: list[int]
    param: Param


def build_benchmark(config: dict[str, Any]) -> BenchmarkData:
    """Build geometry, physics, initialization, and target once for all runs."""
    param = Param()
    for key, value in config["process"].items():
        setattr(param, key, value)

    kernel = Kernel(param)
    domain_dim = list(config["geometry"]["domain_dim"])
    C = jnp.asarray(testcube(domain_dim, list(config["geometry"]["offset"])))
    I_start = C * float(param.I_max) * float(config["initialization"]["scale"])

    layers = jnp.arange(1, kernel.r_z)
    attenuation_sum = jnp.sum(
        jnp.exp(-param.atten_coef * layers * param.layer_height)
    )
    if config["target_energy"]["reference"] != "initial_intensity":
        raise ValueError("Unsupported target energy reference.")
    target_scale = float(config["target_energy"]["scale"])
    T = target_scale * I_start * param.exposure_time * (1.0 + attenuation_sum)

    return BenchmarkData(
        I_start=I_start,
        T=T,
        C=C,
        kernel=kernel,
        domain_dim=domain_dim,
        param=param,
    )


def evaluate_solution(
    benchmark: BenchmarkData,
    optimization: OptimizationResult,
    metric_names: list[str],
) -> tuple[dict[str, Any], jnp.ndarray]:
    """Evaluate a solver result with the same physical metrics for every run."""
    energy = simulate(
        jnp.asarray(optimization.I_opt),
        benchmark.kernel,
        benchmark.domain_dim,
        benchmark.param,
    )
    energy = jax.block_until_ready(energy)
    energy_np = np.asarray(energy, dtype=np.float64)
    C_np = np.asarray(benchmark.C)
    inside_energy = energy_np[C_np == 1]
    outside_energy = energy_np[C_np == 0]
    intensity = np.asarray(optimization.I_opt, dtype=np.float64)

    if inside_energy.size == 0:
        raise ValueError("The benchmark geometry contains no target voxels.")
    if outside_energy.size == 0:
        raise ValueError("The benchmark geometry contains no outside voxels.")

    available = {
        "undercured_voxels": int(
            np.sum(inside_energy < float(benchmark.param.E_crit))
        ),
        "overcured_voxels": int(
            np.sum(outside_energy >= float(benchmark.param.E_crit))
        ),
        "outside_energy": float(np.sum(outside_energy)),
        "min_inside_energy": float(np.min(inside_energy)),
        "max_inside_energy": float(np.max(inside_energy)),
        "min_outside_energy": float(np.min(outside_energy)),
        "max_outside_energy": float(np.max(outside_energy)),
        "mean_inside_energy": float(np.mean(inside_energy)),
        "mean_outside_energy": float(np.mean(outside_energy)),
        "intensity_min": float(np.min(intensity)),
        "intensity_max": float(np.max(intensity)),
        "intensity_mean": float(np.mean(intensity)),
        "runtime_seconds": optimization.runtime_seconds,
        "optimization_success": optimization.success,
        "converged": optimization.converged,
        "iterations": optimization.iterations,
        "gradient_norm": optimization.final_gradient_norm,
    }
    return {name: available[name] for name in metric_names}, energy
