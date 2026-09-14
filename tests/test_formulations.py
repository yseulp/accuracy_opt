import unittest
from types import SimpleNamespace

import jax.numpy as jnp

from scripts.optimization.formulations import FORMULATION_ADAPTERS
from scripts.optimization.objectives import guven_boundary_mask
from scripts.optimization.problems import PROBLEMS


class PenaltyAdapterTest(unittest.TestCase):
    def setUp(self):
        self.C = jnp.array([1.0, 0.0, 0.0])
        self.T = jnp.array([0.4, 0.0, 0.0])
        self.I = jnp.array([0.2, 0.7, 0.1])
        self.kernel = SimpleNamespace(
            weights=jnp.ones((1, 1, 1)),
            r_xy=0,
            r_z=0,
        )
        self.param = SimpleNamespace(
            exposure_time=1.0,
            E_crit=0.5,
            I_max=1.0,
        )
        self.domain_dim = [1, 1, 3]

    def penalty_value(self, problem_name, weight):
        problem = PROBLEMS[problem_name]
        formulation = FORMULATION_ADAPTERS["penalty"].build(
            problem,
            I_start=self.I,
            T=self.T,
            C=self.C,
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters=problem.resolve_parameters({}),
            penalty_parameters={"penalty_weight": weight},
        )
        return float(formulation.objective(self.I))

    def test_guven_penalty_is_added_to_base_objective(self):
        domain_dim = [5, 5, 1]
        C = jnp.zeros(25).at[12].set(1.0)
        I = jnp.linspace(0.0, 0.96, 25).at[12].set(0.2)
        problem = PROBLEMS["guven"]
        formulation = FORMULATION_ADAPTERS["penalty"].build(
            problem,
            I_start=I,
            T=jnp.zeros_like(I),
            C=C,
            kernel=self.kernel,
            domain_dim=domain_dim,
            param=self.param,
            problem_parameters=problem.resolve_parameters({}),
            penalty_parameters={"penalty_weight": 10.0},
        )
        boundary_energy = float(
            jnp.sum(I * guven_boundary_mask(C, domain_dim).astype(I.dtype))
        )
        squared_undercure = (0.5 - 0.2) ** 2

        self.assertAlmostEqual(
            float(formulation.objective(I)),
            boundary_energy + 10.0 * squared_undercure,
        )

    def test_wang_penalty_matches_exterior_penalty_definition(self):
        energy_error = 0.2 + 0.7 + 0.1
        undercure = 0.5 - 0.2
        overcure = 0.7 - 0.5

        self.assertAlmostEqual(
            self.penalty_value("wang", 10.0),
            energy_error + 10.0 * (undercure + overcure),
        )

    def test_reverse_guven_penalty_is_added_to_base_objective(self):
        inside_reward = -0.2
        squared_overcure = (0.7 - 0.5) ** 2

        self.assertAlmostEqual(
            self.penalty_value("reverse_guven", 10.0),
            inside_reward + 10.0 * squared_overcure,
        )


if __name__ == "__main__":
    unittest.main()
