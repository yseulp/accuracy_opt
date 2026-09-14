import unittest
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from scripts.optimization.objectives import (
    obj_fun_2,
    obj_fun_guven,
    obj_fun_reverse_guven,
    obj_fun_wang,
)
from scripts.optimization.solvers import run_solver
from scripts.model.simulation import simulate


class LinprogTest(unittest.TestCase):
    def setUp(self):
        domain_dim = [3, 3, 3]
        C = np.zeros(27, dtype=np.float64)
        C[13] = 1.0
        weights = jnp.zeros((1, 1, 1)).at[0, 0, 0].set(1.0)
        self.domain_dim = domain_dim
        self.C = C
        self.kernel = SimpleNamespace(weights=weights, r_xy=0, r_z=0)
        self.param = SimpleNamespace(exposure_time=1.0, E_crit=0.5, I_max=1.0)
        self.I_start = jnp.asarray(C)

    def run_guven(self, solver, solver_parameters):
        return run_solver(
            solver,
            "guven",
            I_start=self.I_start,
            T=jnp.zeros_like(self.I_start),
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters={},
            solver_parameters=solver_parameters,
        )

    def assert_cure_constraint(self, result):
        energy = np.asarray(
            simulate(
                result.I_opt,
                self.kernel,
                self.domain_dim,
                self.param,
            ),
            dtype=np.float64,
        )
        self.assertTrue(result.success, result.message)
        self.assertGreaterEqual(energy[13], 0.5 - 1e-6)
        self.assertGreaterEqual(float(np.min(result.I_opt)), -1e-7)
        self.assertLessEqual(float(np.max(result.I_opt)), 1.0 + 1e-7)

    def test_guven_linear_program_satisfies_cure_constraint(self):
        result = self.run_guven("linprog_highs", {})
        self.assertEqual(result.formulation, "linear_program")
        self.assert_cure_constraint(result)
        expected_objective = float(
            obj_fun_guven(
                result.I_opt,
                jnp.zeros_like(self.I_start),
                jnp.asarray(self.C),
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )
        self.assertAlmostEqual(result.final_objective, expected_objective, places=8)

    def test_direct_solver_rejects_penalty_parameters(self):
        with self.assertRaisesRegex(ValueError, "does not accept penalty"):
            run_solver(
                "linprog_highs",
                "guven",
                I_start=self.I_start,
                T=jnp.zeros_like(self.I_start),
                C=jnp.asarray(self.C),
                kernel=self.kernel,
                domain_dim=self.domain_dim,
                param=self.param,
                problem_parameters={},
                penalty_parameters={"penalty_weight": 10.0},
                solver_parameters={},
            )

    def test_failed_linprog_reports_objective_for_returned_fallback(self):
        infeasible_param = SimpleNamespace(
            exposure_time=1.0,
            E_crit=2.0,
            I_max=1.0,
        )
        result = run_solver(
            "linprog_highs",
            "guven",
            I_start=self.I_start,
            T=jnp.zeros_like(self.I_start),
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=infeasible_param,
            problem_parameters={},
            solver_parameters={},
        )

        self.assertFalse(result.success)
        np.testing.assert_allclose(result.I_opt, self.I_start)
        self.assertEqual(result.final_objective, result.initial_objective)

    def test_guven_slsqp_satisfies_cure_constraint(self):
        result = self.run_guven(
            "slsqp",
            {"max_iterations": 20, "ftol": 1e-8},
        )
        self.assertEqual(result.formulation, "direct_constraints")
        self.assert_cure_constraint(result)

    def test_guven_trust_constr_satisfies_cure_constraint(self):
        result = self.run_guven(
            "trust_constr",
            {
                "max_iterations": 50,
                "gtol": 1e-6,
                "xtol": 1e-6,
                "barrier_tol": 1e-6,
            },
        )
        self.assertEqual(result.formulation, "linear_program")
        self.assert_cure_constraint(result)

    def assert_wang_solver_matches_original_objective(self, solver):
        target_energy = jnp.zeros_like(self.I_start)
        result = run_solver(
            solver,
            "wang",
            I_start=self.I_start,
            T=target_energy,
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters={},
            solver_parameters=(
                {"max_iterations": 50, "ftol": 1e-8}
                if solver == "slsqp"
                else {}
            ),
        )
        original_objective = float(
            obj_fun_wang(
                result.I_opt,
                target_energy,
                jnp.asarray(self.C),
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.final_objective, original_objective, places=8)
        energy = np.asarray(
            simulate(result.I_opt, self.kernel, self.domain_dim, self.param)
        )
        self.assertGreaterEqual(float(energy[13]), 0.5 - 1e-8)
        self.assertLessEqual(float(np.max(energy[self.C == 0])), 0.5 + 1e-8)

    def test_wang_linear_program_matches_original_objective(self):
        self.assert_wang_solver_matches_original_objective("linprog_highs")

    def test_reverse_guven_slsqp_uses_hard_outside_constraint(self):
        target_energy = jnp.zeros_like(self.I_start)
        result = run_solver(
            "slsqp",
            "reverse_guven",
            I_start=self.I_start,
            T=target_energy,
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters={},
            solver_parameters={"max_iterations": 20, "ftol": 1e-8},
        )
        expected_objective = float(
            obj_fun_reverse_guven(
                result.I_opt,
                target_energy,
                jnp.asarray(self.C),
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.formulation, "direct_constraints")
        self.assertAlmostEqual(result.final_objective, expected_objective, places=8)
        energy = np.asarray(
            simulate(result.I_opt, self.kernel, self.domain_dim, self.param)
        )
        self.assertLessEqual(float(np.max(energy[self.C == 0])), 0.5 + 1e-8)

    def test_reverse_guven_linear_program_uses_hard_outside_constraint(self):
        result = run_solver(
            "linprog_highs",
            "reverse_guven",
            I_start=self.I_start,
            T=jnp.zeros_like(self.I_start),
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters={},
            solver_parameters={},
        )
        energy = np.asarray(
            simulate(result.I_opt, self.kernel, self.domain_dim, self.param)
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(energy[13]), 1.0)
        self.assertLessEqual(float(np.max(energy[self.C == 0])), 0.5 + 1e-8)

    def test_cure_linear_program_matches_soft_violation_objective(self):
        target_energy = jnp.zeros_like(self.I_start)
        result = run_solver(
            "linprog_highs",
            "obj_fun_2",
            I_start=self.I_start,
            T=target_energy,
            C=jnp.asarray(self.C),
            kernel=self.kernel,
            domain_dim=self.domain_dim,
            param=self.param,
            problem_parameters={},
            solver_parameters={},
        )
        expected_objective = float(
            obj_fun_2(
                result.I_opt,
                target_energy,
                jnp.asarray(self.C),
                self.kernel,
                self.domain_dim,
                self.param,
            )
        )

        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(result.final_objective, expected_objective, places=8)


if __name__ == "__main__":
    unittest.main()
