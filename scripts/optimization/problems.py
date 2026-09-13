"""Solver-independent definitions of the mathematical optimization problems."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from scripts.model.simulation import simulate
from scripts.optimization.objectives import (
    obj_fun_1,
    obj_fun_2,
    obj_fun_guven,
    obj_fun_reverse_guven,
    obj_fun_wang,
)


class OptimizationProblem:
    name: str
    objective_is_smooth = False
    has_constraints = False

    def bounds(self, n_variables: int, param) -> list[tuple[float, float]]:
        return [(0.0, float(param.I_max)) for _ in range(n_variables)]

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        raise NotImplementedError

    def constraint_residuals(
        self,
        I,
        T,
        C,
        kernel,
        domain_dim,
        param,
        problem_parameters,
    ):
        del I, T, C, kernel, domain_dim, param, problem_parameters
        return jnp.empty((0,))

    def parameter_defaults(self) -> dict[str, Any]:
        return {}

    def resolve_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        defaults = self.parameter_defaults()
        unknown = set(parameters) - set(defaults)
        if unknown:
            raise ValueError(
                f"Problem {self.name!r} does not use parameter(s): "
                + ", ".join(sorted(unknown))
            )
        return {**defaults, **parameters}


class EnergyProblem(OptimizationProblem):
    name = "obj_fun_1"
    objective_is_smooth = True

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        del problem_parameters
        return obj_fun_1(I, T, C, kernel, domain_dim, param)


class CureProblem(OptimizationProblem):
    name = "obj_fun_2"
    objective_is_smooth = False

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        del problem_parameters
        return obj_fun_2(I, T, C, kernel, domain_dim, param)


class GuvenProblem(OptimizationProblem):
    name = "guven"
    objective_is_smooth = True
    has_constraints = True

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        return obj_fun_guven(I, T, C, kernel, domain_dim, param)

    def constraint_residuals(
        self,
        I,
        T,
        C,
        kernel,
        domain_dim,
        param,
        problem_parameters,
    ):
        del T, problem_parameters
        E = simulate(I, kernel, domain_dim, param)
        return E[C == 1] - float(param.E_crit)


class WangProblem(OptimizationProblem):
    name = "wang"
    objective_is_smooth = False
    has_constraints = True

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        del problem_parameters
        return obj_fun_wang(I, T, C, kernel, domain_dim, param)

    def constraint_residuals(
        self,
        I,
        T,
        C,
        kernel,
        domain_dim,
        param,
        problem_parameters,
    ):
        del T, problem_parameters
        E = simulate(I, kernel, domain_dim, param)
        return jnp.concatenate(
            (
                E[C == 1] - float(param.E_crit),
                float(param.E_crit) - E[C == 0],
            )
        )


class ReverseGuvenProblem(OptimizationProblem):
    name = "reverse_guven"
    objective_is_smooth = True
    has_constraints = True

    def objective(self, I, T, C, kernel, domain_dim, param, problem_parameters):
        del problem_parameters
        return obj_fun_reverse_guven(I, T, C, kernel, domain_dim, param)

    def constraint_residuals(
        self,
        I,
        T,
        C,
        kernel,
        domain_dim,
        param,
        problem_parameters,
    ):
        del T, problem_parameters
        E = simulate(I, kernel, domain_dim, param)
        return float(param.E_crit) - E[C == 0]


PROBLEMS: dict[str, OptimizationProblem] = {
    problem.name: problem
    for problem in (
        EnergyProblem(),
        CureProblem(),
        GuvenProblem(),
        WangProblem(),
        ReverseGuvenProblem(),
    )
}
