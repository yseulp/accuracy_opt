import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from scripts.model.simulation import simulate
from scripts.optimization.formulations import (
    FORMULATION_ADAPTERS,
    build_energy_matrix,
)
from scripts.optimization.objectives import (
    guven_boundary_mask,
    obj_fun_guven,
)
from scripts.optimization.problems import PROBLEMS


class GuvenBoundaryTest(unittest.TestCase):
    def setUp(self):
        self.domain_dim = [7, 7, 3]
        target = np.zeros(self.domain_dim, dtype=np.float64)
        target[3, 3, 1] = 1.0
        self.C = jnp.asarray(target.ravel(order="C"))

        expected_boundary = np.zeros(self.domain_dim, dtype=bool)
        expected_boundary[2:5, 2:5, 1] = True
        expected_boundary[3, 3, 1] = False
        self.expected_boundary = expected_boundary

        self.kernel = SimpleNamespace(
            weights=jnp.ones((1, 1, 1)),
            r_xy=0,
            r_z=0,
        )
        self.param = SimpleNamespace(
            exposure_time=1.0,
            E_crit=0.5,
            I_max=1000.0,
        )
        self.I = jnp.arange(np.prod(self.domain_dim), dtype=jnp.float64)

    def boundary(self):
        return np.asarray(
            guven_boundary_mask(self.C, self.domain_dim), dtype=bool
        ).reshape(self.domain_dim, order="C")

    def test_boundary_is_the_exterior_xy_neighbor_shell(self):
        boundary = self.boundary()
        target = np.asarray(self.C, dtype=bool).reshape(
            self.domain_dim, order="C"
        )

        self.assertFalse(np.any(boundary[target]))
        np.testing.assert_array_equal(boundary, self.expected_boundary)
        self.assertFalse(boundary[0, 0, 1])
        self.assertTrue(boundary[2, 2, 1])
        self.assertTrue(boundary[2, 3, 1])
        self.assertTrue(boundary[3, 2, 1])

    def test_boundary_does_not_expand_into_adjacent_z_layers(self):
        boundary = self.boundary()

        self.assertFalse(np.any(boundary[:, :, 0]))
        self.assertFalse(np.any(boundary[:, :, 2]))

    def test_objective_is_exactly_the_boundary_energy(self):
        energy = simulate(
            self.I,
            self.kernel,
            self.domain_dim,
            self.param,
        )
        expected = float(
            np.sum(
                np.asarray(energy).reshape(self.domain_dim, order="C")[
                    self.expected_boundary
                ]
            )
        )
        actual = float(
            obj_fun_guven(
                self.I,
                jnp.zeros_like(self.I),
                self.C,
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )

        self.assertAlmostEqual(actual, expected)

    def test_energy_far_from_the_boundary_does_not_change_objective(self):
        far_index = np.ravel_multi_index((0, 0, 1), self.domain_dim, order="C")
        changed_intensity = self.I.at[far_index].add(10000.0)
        objective = lambda intensity: obj_fun_guven(
            intensity,
            jnp.zeros_like(intensity),
            self.C,
            self.kernel,
            self.domain_dim,
            self.param,
        )

        self.assertAlmostEqual(
            float(objective(self.I)),
            float(objective(changed_intensity)),
        )

    def test_objective_remains_jax_differentiable(self):
        value, gradient = jax.value_and_grad(
            lambda intensity: obj_fun_guven(
                intensity,
                jnp.zeros_like(intensity),
                self.C,
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )(self.I)

        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.all(np.isfinite(np.asarray(gradient))))

    def test_linear_formulation_matches_boundary_objective(self):
        problem = PROBLEMS["guven"]
        formulation = FORMULATION_ADAPTERS["linear"].build(
            problem,
            I_start=self.I,
            T=jnp.zeros_like(self.I),
            C=self.C,
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters=problem.resolve_parameters({}),
            penalty_parameters={},
        )
        energy_matrix = build_energy_matrix(
            self.kernel,
            self.domain_dim,
            self.param,
        )
        boundary_indices = np.flatnonzero(
            self.expected_boundary.ravel(order="C")
        )
        expected_coefficients = np.asarray(
            energy_matrix[boundary_indices].sum(axis=0)
        ).ravel()
        expected_objective = float(
            obj_fun_guven(
                self.I,
                jnp.zeros_like(self.I),
                self.C,
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )

        np.testing.assert_allclose(formulation.c, expected_coefficients)
        self.assertAlmostEqual(
            float(np.dot(formulation.c, np.asarray(self.I))),
            expected_objective,
        )

    def test_guven_constraint_remains_inside_cure_requirement(self):
        problem = PROBLEMS["guven"]
        energy = simulate(
            self.I,
            self.kernel,
            self.domain_dim,
            self.param,
        )
        residuals = problem.constraint_residuals(
            self.I,
            jnp.zeros_like(self.I),
            self.C,
            self.kernel,
            self.domain_dim,
            self.param,
            {},
        )

        np.testing.assert_allclose(
            residuals,
            energy[self.C == 1] - self.param.E_crit,
        )
        self.assertGreaterEqual(float(residuals[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
