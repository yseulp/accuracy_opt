import unittest

import jax.numpy as jnp
import numpy as np

from scripts.optimization.formulations import (
    ConstrainedFormulation,
    LinearConstraintSpec,
    ObjectiveFormulation,
)
from scripts.optimization.solvers import (
    solve_pgd,
    solve_slsqp_constrained,
    solve_slsqp_objective,
    solve_trust_constr_objective,
)


class ProjectedGradientDescentTest(unittest.TestCase):
    def test_tolerance_compares_values_after_updates(self):
        result = solve_pgd(
            ObjectiveFormulation(
                objective=lambda intensity: jnp.sum((intensity - 1.0) ** 2),
                bounds=[(0.0, 2.0)],
            ),
            I_start=jnp.array([0.0]),
            solver_parameters={
                "step_size": 0.1,
                "max_iterations": 5,
                "objective_tolerance": 0.0,
            },
        )

        self.assertFalse(result.success)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 5)
        self.assertLess(result.final_objective, result.initial_objective)


class SlsqpConstraintTest(unittest.TestCase):
    def test_accepts_matrix_based_linear_constraint(self):
        formulation = ConstrainedFormulation(
            objective=lambda x: float((x[0] - 2.0) ** 2),
            objective_grad=lambda x: np.array([2.0 * (x[0] - 2.0)]),
            constraints=(
                LinearConstraintSpec(
                    matrix=np.array([[1.0]]),
                    lower_bound=-np.inf,
                    upper_bound=1.0,
                ),
            ),
            bounds=[(0.0, 3.0)],
        )

        result = solve_slsqp_constrained(
            formulation,
            I_start=np.array([0.0]),
            solver_parameters={"max_iterations": 20, "ftol": 1e-9},
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(result.I_opt[0]), 1.0, places=7)

    def test_slsqp_accepts_smooth_objective(self):
        result = solve_slsqp_objective(
            ObjectiveFormulation(
                objective=lambda intensity: jnp.sum((intensity - 1.0) ** 2),
                bounds=[(0.0, 2.0)],
            ),
            I_start=np.array([0.0]),
            solver_parameters={"max_iterations": 20, "ftol": 1e-9},
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(result.I_opt[0]), 1.0, places=7)

    def test_trust_constr_accepts_smooth_objective(self):
        result = solve_trust_constr_objective(
            ObjectiveFormulation(
                objective=lambda intensity: jnp.sum((intensity - 1.0) ** 2),
                bounds=[(0.0, 2.0)],
            ),
            I_start=np.array([0.0]),
            solver_parameters={
                "max_iterations": 50,
                "gtol": 1e-8,
                "xtol": 1e-8,
                "barrier_tol": 1e-8,
            },
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(result.I_opt[0]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
