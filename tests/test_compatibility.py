import unittest

from scripts.optimization.problems import OptimizationProblem, PROBLEMS
from scripts.optimization.solvers import (
    FormulationHandler,
    SOLVERS,
    SolverDefinition,
    check_compatibility,
    resolve_selection,
)


class CompatibilityTest(unittest.TestCase):
    def test_supported_combinations(self):
        self.assertEqual(
            check_compatibility("obj_fun_1", "projected_gradient_descent"),
            "objective",
        )
        self.assertEqual(
            check_compatibility("obj_fun_2", "projected_gradient_descent"),
            "objective",
        )
        for problem in ("guven", "wang", "reverse_guven"):
            self.assertEqual(
                check_compatibility(problem, "projected_gradient_descent"),
                "penalty",
            )
        self.assertEqual(check_compatibility("guven", "slsqp"), "constrained")
        self.assertEqual(
            check_compatibility("reverse_guven", "slsqp"), "constrained"
        )
        self.assertEqual(check_compatibility("obj_fun_1", "slsqp"), "objective")
        self.assertEqual(check_compatibility("obj_fun_1", "trust_constr"), "objective")
        self.assertEqual(check_compatibility("obj_fun_2", "trust_constr"), "linear")
        self.assertEqual(check_compatibility("guven", "trust_constr"), "linear")
        self.assertEqual(check_compatibility("reverse_guven", "trust_constr"), "linear")
        self.assertEqual(check_compatibility("guven", "linprog_highs"), "linear")
        self.assertEqual(check_compatibility("wang", "trust_constr"), "linear")
        self.assertEqual(check_compatibility("wang", "linprog_highs"), "linear")
        self.assertEqual(check_compatibility("obj_fun_2", "linprog_highs"), "linear")
        self.assertEqual(
            check_compatibility("reverse_guven", "linprog_highs"), "linear"
        )

    def test_incompatible_combination_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Incompatible combination"):
            check_compatibility("obj_fun_1", "linprog_highs")
        with self.assertRaisesRegex(ValueError, "requires a smooth objective"):
            check_compatibility("obj_fun_2", "slsqp")
        with self.assertRaisesRegex(ValueError, "requires a smooth formulation"):
            check_compatibility("wang", "slsqp")

    def test_irrelevant_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not accept penalty"):
            resolve_selection(
                "guven",
                "linprog_highs",
                penalty_parameters={"penalty_weight": 100.0},
            )
        with self.assertRaisesRegex(ValueError, "does not use parameter"):
            resolve_selection(
                "guven",
                "linprog_highs",
                solver_parameters={"step_size": 0.03},
            )

    def test_solver_aliases_are_normalized(self):
        resolved = resolve_selection("guven", "linprog")
        self.assertEqual(resolved["solver"], "linprog_highs")
        self.assertEqual(resolved["formulation"], "linear_program")
        self.assertEqual(resolved["constraint_handling"], "direct")

    def test_new_problem_uses_existing_solver_without_dispatch_change(self):
        class NewObjectiveProblem(OptimizationProblem):
            name = "new_objective"
            objective_is_smooth = True

        PROBLEMS[NewObjectiveProblem.name] = NewObjectiveProblem()
        try:
            resolved = resolve_selection(
                "new_objective",
                "projected_gradient_descent",
            )
        finally:
            del PROBLEMS[NewObjectiveProblem.name]

        self.assertEqual(resolved["formulation"], "base_objective")

    def test_new_solver_is_fully_described_by_registry_entry(self):
        SOLVERS["new_linear_solver"] = SolverDefinition(
            formulation_handlers={
                "linear": FormulationHandler(
                    solve=lambda *args, **kwargs: None,
                )
            },
            parameter_defaults={},
        )
        try:
            resolved = resolve_selection("guven", "new_linear_solver")
        finally:
            del SOLVERS["new_linear_solver"]

        self.assertEqual(resolved["formulation"], "linear_program")

    def test_parameters_follow_selected_formulation(self):
        pgd = resolve_selection("guven", "projected_gradient_descent")
        linear = resolve_selection("guven", "linprog")

        self.assertEqual(pgd["problem_parameters"], {})
        self.assertEqual(pgd["penalty_parameters"], {"penalty_weight": 100.0})
        self.assertIn("step_size", pgd["solver_parameters"])
        self.assertEqual(linear["problem_parameters"], {})
        self.assertEqual(linear["penalty_parameters"], {})
        self.assertEqual(linear["solver_parameters"], {})

        cure_linear = resolve_selection("obj_fun_2", "linprog")
        self.assertEqual(cure_linear["constraint_handling"], "none")


if __name__ == "__main__":
    unittest.main()
